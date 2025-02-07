import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

image_path = "Bilder/182-Wir-bauen-Bayern02-2-1536x960.jpeg"
image = Image.open(image_path)

# conditional image captioning
# text = "a photography of"
# inputs = processor(image, text, return_tensors="pt").to("cuda", torch.float16)

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
