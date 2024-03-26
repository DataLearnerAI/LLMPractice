import gradio as gr
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

bot_avatar = "/home/datalearner/dl_logo_rect.png"
user_avatar = "/home/datalearner/user_avatar.png"

def init_model():
    path = '/home/datalearner/MiniCPM-2B-dpo-fp16'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return model, streamer, tokenizer


with gr.Blocks() as demo:
    mini_cpm_model, mini_cpm_streamer, mini_cpm_tokenizer = init_model()
    chatbot = gr.Chatbot(
        height=900,
        avatar_images=(user_avatar, bot_avatar)
        )
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    cpm_history = []

    def clear_history():
        global cpm_history
        cpm_history = []

    def respond(message, chat_history):
        
        cpm_history.append({"role": "user", "content": message})
        history_str = mini_cpm_tokenizer.apply_chat_template(cpm_history, tokenize=False, add_generation_prompt=False)
        inputs = mini_cpm_tokenizer(history_str, return_tensors='pt').to("cuda")

        chat_history.append([message, ""])
        generation_kwargs = dict(**inputs, streamer=mini_cpm_streamer, max_new_tokens=4096, pad_token_id=mini_cpm_tokenizer.eos_token_id, num_beams=1, do_sample=True, top_p=0.8,
                    temperature=0.3)
        thread = Thread(target=mini_cpm_model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in mini_cpm_streamer:
            chat_history[-1][1] += new_text
            yield "", chat_history
        
        cpm_history.append({"role": "assistant", "content": chat_history[-1][1]})
        

    clear.click(clear_history)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=80)