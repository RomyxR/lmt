import time
import os
import json

def models_list():
    def format_size(size_bytes):
        if size_bytes == 0: return "0 B"
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0: return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
    
    models_path = "models"
    for file in os.listdir(models_path):
        if file.endswith(".gguf"):
            full_path = os.path.join(models_path, file)
            size = format_size(os.path.getsize(full_path))
            print(file, size)

def llamacpp_chat(model_path: str, system_prompt: str = ""):
    from llama_cpp import Llama
    
    if not os.path.exists(model_path):
        print("Модель не найдена!")
        return
    
    llm = Llama(
        model_path=model_path,
        n_ctx=32768,
        n_batch=512,
        verbose=False,
        n_gpu_layers=-1,
        logits_all=False,
        swa_full=False
    )

    memory = []
    if system_prompt.strip():
        memory.append({"role": "system", "content": system_prompt})
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input == "cls":
                memory = []
                if system_prompt.strip():
                    memory.append({"role": "system", "content": system_prompt})
                print("Память удалена!")
                continue
            if user_input == "savechat":
                with open(f"chat{str(time.time()).replace('.', '')}.json", "w") as f:
                    json.dump(memory, f, ensure_ascii=False)
                continue
            
            memory.append({"role": "user", "content": user_input})

            start_time = time.time()

            stream = llm.create_chat_completion(
                messages=memory,
                max_tokens=1024,
                temperature=0.6,
                top_p=0.95,
                repeat_penalty=1.2,
                stream=True,
            )

            full_response = ""
            for chunk in stream:
                content = chunk['choices'][0]['delta'].get('content')
                if content:
                    print(content, end="", flush=True)
                    full_response += content

            print()
            print(f"[{time.time() - start_time:.2f} sec]")

            memory.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\nВыход...")

