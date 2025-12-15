"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from ytch.device import get_sensible_device

from avp_vit import AVPConfig


@dataclass
class ScheduleEntry:
    """A single stage in the training schedule.

    Each stage has: lr_ramp_steps of 0→1 ramp, then decay 1→0 for remainder.
    """

    phase: Literal["probe", "main"]
    grid_size: int
    start_step: int
    end_step: int
    lr_ramp_steps: int

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step + 1

    @property
    def lr_decay_steps(self) -> int:
        return self.duration - self.lr_ramp_steps


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    ckpt_dir: Path = Path("checkpoints")
    # Model
    avp: AVPConfig = field(
        default_factory=lambda: AVPConfig(
            scene_grid_size=64,  # Max grid size for curriculum
            glimpse_grid_size=7,
            layer_scale_init=1e-2,
            use_output_proj=True,
            n_scene_registers=32,
            gradient_checkpointing=True,
            use_convex_gating=True,
            use_local_temporal=True,
        )
    )
    freeze_inner_backbone: bool = False
    # Curriculum (small → large for main training)
    grid_sizes: tuple[int, ...] = (16, 32, 64)
    # Training
    n_viewpoints_per_step: int = 2  # Inner loop viewpoints (>=2 for length generalization)
    n_steps: int = 200000
    batch_size: int = 32  # Max batch size (at max grid size)
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    probe_ratio: float = 0.01  # Fraction of n_steps for probe phase (all sizes, largest first)
    probe_ramp_ratio: float = 0.5  # Within each probe stage, fraction that's LR ramp
    main_ramp_steps: int = 500  # Within each main stage, fixed LR ramp steps
    grad_clip: float = 1.0
    crop_scale_min: float = 0.4
    loss: Literal["l1", "mse"] = "mse"
    # Logging
    log_every: int = 20
    val_every: int = 50
    ckpt_every: int = 500
    # Compilation
    compile: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def max_grid_size(self) -> int:
        return max(self.grid_sizes)

    @property
    def probe_steps(self) -> int:
        return int(self.n_steps * self.probe_ratio)

    @property
    def probe_steps_per_size(self) -> int:
        return self.probe_steps // len(self.grid_sizes)

    @property
    def main_steps(self) -> int:
        return self.n_steps - self.probe_steps

    @property
    def main_steps_per_stage(self) -> int:
        return self.main_steps // len(self.grid_sizes)

    @property
    def probe_grid_sizes(self) -> tuple[int, ...]:
        """Largest first for OOM detection."""
        return tuple(reversed(self.grid_sizes))

    def get_schedule(self) -> list[ScheduleEntry]:
        """Return full training schedule."""
        schedule: list[ScheduleEntry] = []

        # Probe phase (largest → smallest)
        step = 0
        probe_sizes = self.probe_grid_sizes
        for i, G in enumerate(probe_sizes):
            end = step + self.probe_steps_per_size - 1
            if i == len(probe_sizes) - 1:
                end = self.probe_steps - 1  # Last stage absorbs remainder
            duration = end - step + 1
            assert duration > 0, f"Probe stage G={G} has non-positive duration"
            ramp = int(duration * self.probe_ramp_ratio)
            assert 0 < ramp < duration, f"Probe stage G={G}: ramp={ramp} invalid for duration={duration}"
            schedule.append(ScheduleEntry(
                phase="probe",
                grid_size=G,
                start_step=step,
                end_step=end,
                lr_ramp_steps=ramp,
            ))
            step = end + 1

        assert step == self.probe_steps, f"Probe phase ends at {step}, expected {self.probe_steps}"

        # Main phase (smallest → largest)
        for i, G in enumerate(self.grid_sizes):
            start = self.probe_steps + i * self.main_steps_per_stage
            end = start + self.main_steps_per_stage - 1
            if i == len(self.grid_sizes) - 1:
                end = self.n_steps - 1
            duration = end - start + 1
            assert duration > self.main_ramp_steps, f"Main stage G={G}: duration={duration} < ramp={self.main_ramp_steps}"
            schedule.append(ScheduleEntry(
                phase="main",
                grid_size=G,
                start_step=start,
                end_step=end,
                lr_ramp_steps=self.main_ramp_steps,
            ))

        assert schedule[-1].end_step == self.n_steps - 1, f"Schedule ends at {schedule[-1].end_step}, expected {self.n_steps - 1}"
        return schedule
