import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def image_captioning(img_url):
    """
    Generate image captions using BLIP model.

    Args:
        img_url (str): Path to the image file.

    Returns:
        str: Generated caption for the image.
    """
    # Load the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
    ).to("cuda")

    # Open and process the image
    raw_image = Image.open(img_url).convert("RGB")

    # Conditional image captioning
    # text = "This is "
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
    # Load the processor and model

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
    ).to("cuda")

    raw_image = Image.open(img_url).convert("RGB")

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))


# >>> a woma`n sitting on the beach with her dog
if __name__ == "__main__":
    img_url = (
        "/home/zhihao/cs272project/data/input/cam_0001_0000 copy.jpg"  ,# 本地图片路径
    )

    caption = image_captioning(img_url)
    print(f"Generated Caption: {caption}")
