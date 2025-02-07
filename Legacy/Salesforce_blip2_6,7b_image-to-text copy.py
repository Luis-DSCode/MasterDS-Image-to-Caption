from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b-coco", device_map="auto"
)  # doctest: +IGNORE_RESULT

image_path = "Bilder/184-InsureMe_155301-600x450.jpg"
image = Image.open(image_path)

if image.mode != "RGB":
    image = image.convert(mode="RGB")

prompt = "Describe what can be seen in this image so it can easily be searched. The name of this pictue is " + os.path.basename(image_path)

inputs = processor(images=image, text=prompt ,return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=256,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)