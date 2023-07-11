import os
import torch
import platform
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from utils import (
    load_pretrained,
    prepare_infer_args
)

def init_model():
    print("init model ...")
    
    model_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    model.generation_config = GenerationConfig.from_pretrained(
        model_args.model_name_or_path
    )
    model.generation_config.max_new_tokens = generating_args.max_new_tokens
    model.generation_config.temperature = generating_args.temperature
    model.generation_config.top_k = generating_args.top_k
    model.generation_config.top_p = generating_args.top_p
    model.generation_config.repetition_penalty = generating_args.repetition_penalty
    model.generation_config.do_sample = generating_args.do_sample
    model.generation_config.num_beams = generating_args.num_beams
    model.generation_config.length_penalty = generating_args.length_penalty
    
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，clear 清空历史，CTRL+C 中断生成，steam 开关流式生成，exit 结束。")
    return []


def main(stream=True):
    model, tokenizer = init_model()

    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        print(Fore.CYAN + Style.BRIGHT + "\nBaichuan：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
        messages.append({"role": "assistant", "content": response})

    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
