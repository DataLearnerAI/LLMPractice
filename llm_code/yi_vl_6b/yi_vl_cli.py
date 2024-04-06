import gradio as gr
import os

import torch
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    load_pretrained_model,
    process_images,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image


def load_image(image_file):
    if image_file is None:
        return None
    
    image = Image.open(image_file).convert("RGB")
    return image


bot_avatar = "/home/datalearner/dl_logo_rect.png"
user_avatar = "/home/datalearner/user_avatar.png"
model_path = "/home/datalearner/Yi-VL-6B"

client = None
conv = None
model = None
tokenizer = None
image_processor = None
image_tensor = None
yi_chat_history = []

def add_message(history, message):
    """
    处理输入的消息
    :param history: 历史消息，列表格式，即["msg1", "msg2"]
    :param message: 当前输入的消息，字典类型，其中可以一次放入多个文件。格式如下：
                     {
                        'text': 'hello',
                        'files': [
                            {
                                'path': 'C:\\test.png',
                                'url': '/file=C:\\test.png',
                                'size': 27599, 'orig_name': 'test.png', 
                                'mime_type': 'image/png',
                                'is_stream': False,
                                'meta': {'_type': 'gradio.FileData'}
                            }
                            ]
    """
    global yi_chat_history
    new_msg = {
        "role": "user",
        "content" : {
            "image_files":[],
            "text_msg": "",
        }
    }

    image_file = []

    for x in message["files"]:
        history.append(((x,), None))  
        image_file.append(x)

    if message["text"] is not None:
        history.append((message["text"], None))
        new_msg["content"]["text_msg"] = message["text"]

    if len(image_file)>0:
        new_msg["content"]["image_files"].extend(image_file)

    yi_chat_history.append(new_msg)
    
    return history, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])

def bot(history):
    
    global yi_chat_history
    print(f"chat history: {yi_chat_history}")
    new_text_msg, new_image_list = yi_chat_history[-1]["content"]["text_msg"], yi_chat_history[-1]["content"]["image_files"]

    new_image_file = None
    if len(new_image_list)>0:
        new_image_file = new_image_list[0]

    bot_msg = get_res_from_yi_vl(new_text_msg, new_image_file)
    history[-1][1] = bot_msg
    yi_chat_history.append({"role":"assistant", "content": bot_msg})
    return history


def get_res_from_yi_vl(new_msg, image_file):
    """
    永远只取最新的一个图片作为输入
    """

    print(f"new msg:{new_msg}")
    print(f"new image:{image_file}")

    global conv, model, tokenizer, image_processor, image_tensor

    if image_file is None:
        image = None
    else:
        image = load_image(image_file)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

    inp = new_msg
    if image is not None:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
    else:
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)
    )
    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    print(outputs)

    conv.messages[-1][-1] = outputs
    return outputs


def init_model():
    global tokenizer, model, image_processor, conv, model_path
    model_path = os.path.expanduser(model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path)

    conv = conv_templates["mm_default"].copy()


with gr.Blocks() as demo:
    
    init_model()

    def clear_history():
        global yi_chat_history, conv
        yi_chat_history = []
        conv = conv_templates["mm_default"].copy()

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=900,
        avatar_images=(user_avatar, bot_avatar)
    )
    

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="请输入文本或者上传图片", show_label=False)
    clear = gr.ClearButton([chat_input, chatbot])
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    chat_msg.then(lambda: gr.Textbox(interactive=True), None, [chat_input], queue=False)

    clear.click(clear_history)

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=80)
