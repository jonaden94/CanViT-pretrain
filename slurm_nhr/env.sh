# Grete job setup. Sources .envrc for env vars, sets GWDG proxy for internet access.
# Assumes working directory is repo root (SLURM default = submission dir).

# GWDG proxy — required for HuggingFace Hub (teacher model download)
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export https_proxy="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export http_proxy="http://www-cache.gwdg.de:3128"

source .envrc.grete

echo "[env] Done"
