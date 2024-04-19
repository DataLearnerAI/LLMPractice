import torch
import gradio as gr
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
device = "cuda" # the device to load the model onto


bot_avatar = "/home/datalearner/dl_logo_rect.png"
user_avatar = "/home/datalearner/user_avatar.png"
model_path = "/home/datalearner/Meta-Llama-3-8B-Instruct"
llama3_chat_history = [{"role": "system", "content": "You are a helpful assistant trained by MetaAI! But you are running with DataLearnerAI Code."}]
tokenizer = None
streamer = None
model = None
terminators = None

def init_model():
    global tokenizer, model, streamer, terminators
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)
    
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


with gr.Blocks() as demo:
    init_model()

    chatbot = gr.Chatbot(
        height=900,
        avatar_images=(user_avatar, bot_avatar)
        )
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def clear_history():
        global llama3_chat_history
        llama3_chat_history = []

    def respond(message, chat_history):

        global llama3_chat_history, tokenizer, model, streamer
        
        llama3_chat_history.append({"role": "user", "content": message})
        history_str = tokenizer.apply_chat_template(llama3_chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(history_str, return_tensors='pt').to("cuda")

        chat_history.append([message, ""])
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=4096, pad_token_id=tokenizer.eos_token_id, num_beams=1, do_sample=True, top_p=0.8,
                    temperature=0.3, eos_token_id=terminators)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            chat_history[-1][1] += new_text
            yield "", chat_history
        
        llama3_chat_history.append({"role": "assistant", "content": chat_history[-1][1]})
        

    clear.click(clear_history)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=80)

