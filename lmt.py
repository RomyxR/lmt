import model_chat
import model_download
import sys

if len(sys.argv) < 2:
    print("Вводи аргументы, а не тупи.") 
    sys.exit(1)

match sys.argv[1]:
    case 'run':
        sys_prompt = sys.argv[3] if len(sys.argv) > 3 else ""
        model_chat.llamacpp_chat(sys.argv[2], sys_prompt)
    case 'pull':
        model_download.llamacpp_model_download(sys.argv[2])
    case 'list':
        model_chat.models_list()
    case _: print("Нет какой команды.")