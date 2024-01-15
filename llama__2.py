from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM,TextStreamer
from ctransformers import AutoModelForCausalLM
import time
import sys

import colorama

from colorama import Fore,Style,Back
colorama.init()

# check ctransformers doc for more configs
config = {'max_new_tokens': 200, 'repetition_penalty': 1.1, 
          'temperature': 0.1, 'stream': True}

llmss = AutoModelForCausalLM.from_pretrained(
      r"D:\GENAI\llama", 
      model_type="llama",                                           
      #lib='avx2', for cpu use
      gpu_layers=110, #110 for 7b, 130 for 13b
      **config
      )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print("\n")
print(" \033[38;2;255;165;0m Meet our intelligent chatbot LLAMA2! Ask me anything, and I'll do my best to provide helpful information. \033[0m")


conversation = []

def update_chat(conversation,role,content):
    conve = conversation.append({"role":role,"content":content})

magenta = "\x1b[1;35;40m"

while True:

    print('\n')
    
    print(f"\n{magenta}  🙋 ==>>  ", end='')
    user = input()
    
    
    print("\n")
    update_chat(conversation,'user',user)
    # print(conversation)

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")

    streamer = TextStreamer(input_ids)

    K = llmss.generate(input_ids[0])



    
    def stream_text(text, typing_speed=0.02):
        org = ""
        sys.stdout.write("\033[93m" +" 🤖 ==>>  "+ "\033[0m")
        for char in K:

            text = tokenizer.decode(char)+" "
            org+=text
            sys.stdout.write("\033[93m" + text+ "\033[0m")
            sys.stdout.flush()
            time.sleep(typing_speed)
            
        return org
# text_to_stream = "This is a streaming text example. Enjoy the typing effect!"

    txt = stream_text(K)
    update_chat(conversation,"assistant",txt)
    print("\n")
    print("-"*155)


print("Thank You !")

