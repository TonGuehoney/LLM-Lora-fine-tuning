
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download


snapshot_download(repo_id="01-ai/Yi-6B",
                  repo_type="model",
                  local_dir="J:/xyj/Yi-6B",
                  resume_download=True)



# 01-ai/Yi-6B-200K
# 01-ai/Yi-34B