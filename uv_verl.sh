
set -ex
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_verl --python 3.10
source uv_verl/bin/activate
export UV_LINK_MODE=copy
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
uv pip install --no-deps -e .
uv pip install --upgrade setuptools
uv pip install math-verify[antlr4_9_3]
uv pip install peft==0.15.2 transformers==4.51.3