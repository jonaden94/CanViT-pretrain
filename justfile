# CanViT-eval justfile

# Run all checks
check: lint typecheck test

# Run linter
lint:
    uv run ruff check canvit_eval

# Run type checker
typecheck:
    uv run basedpyright canvit_eval

# Run tests
test:
    uv run pytest -v

# Run tests with coverage
test-cov:
    uv run pytest --cov=canvit_eval --cov-report=term-missing

# Format code
fmt:
    uv run ruff format canvit_eval
    uv run ruff check --fix canvit_eval

# Run IN1k evaluation
in1k val_dir model_repo probe_repo:
    uv run python -m canvit_eval.in1k --val-dir {{val_dir}} --model-repo {{model_repo}} --probe-repo {{probe_repo}}

# Run ADE20K probe training
ade20k model_repo ade20k_root:
    uv run python -m canvit_eval.ade20k --model-repo {{model_repo}} --ade20k-root {{ade20k_root}}
