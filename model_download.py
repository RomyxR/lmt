from huggingface_hub import hf_hub_download
import os
import json

def llamacpp_model_download(model: str):
    with open("lmtconf.json", 'r', encoding="utf-8") as f:
        models_path = json.loads(f.read())["model_path"]
    if ":" not in model:
        print("Mодель должна быть в формате repo_id:filename")
        return
    os.makedirs("models", exist_ok=True)
    repo_id, filename = model.split(":", 1)  # split only once
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_path,
        )
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")

