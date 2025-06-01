from openai import OpenAI

from PIL import Image
import io
import base64


def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format="webp")
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode("utf-8")
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None


input_image_path = "/home/zhihao/cs272project/data/output/Nvi.png"
prompt = "这个人在做什么？请用中文回答。"

base64_image = convert_image_to_webp_base64(input_image_path)
client = OpenAI(
    base_url="https://api-inference.modelscope.cn/v1/",
    api_key="14a623da-3226-41dc-bd0e-f0d439662565",  # ModelScope Token
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct",  # ModelScope Model-Id
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
