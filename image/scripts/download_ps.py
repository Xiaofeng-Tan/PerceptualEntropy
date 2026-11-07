import os
from huggingface_hub import login, snapshot_download

# 登录（请设置环境变量 HF_TOKEN 或替换为你的 Hugging Face token）
login(token=os.environ.get("HF_TOKEN", "your_hf_token_here"))

repo_id = "yuvalkirstain/PickScore_v1"

local_dir = snapshot_download(
    repo_id=repo_id,
    local_dir="../pretrained_ckpt/ps",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Download complete: {local_dir}")