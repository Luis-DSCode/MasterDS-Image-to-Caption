from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

input = "Bilder/184-InsureMe_155301-600x450.jpg"
image = Image.open(input)

if image.mode != "RGB":
    image = image.convert(mode="RGB")

prompt = (
    f"Describe the unique elements in this image titled '{os.path.basename(input)}'. "
    "Provide distinct descriptive details to make it searchable. "
    "Consider specifics in the scene, including any actions or settings."
)

inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.float16)


out = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=256,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id,
        repetition_penalty=1.2
    )

print(processor.decode(out[0], skip_special_tokens=True).strip())