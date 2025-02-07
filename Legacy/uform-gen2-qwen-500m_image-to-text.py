from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
import os
from PIL.ExifTags import TAGS
from datetime import datetime

from extract_metadata import *
from image_to_text_face_recognition import *

def image_to_text_description(input, reference_folder):
    reference_faces = load_reference_faces(reference_folder)
    list_of_faces = generate_string_of_faces(input, reference_faces)

    model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

    prompt = (
        f"Generate a short description for the image titled '{os.path.basename(input)}'. "
        f"Focus on the main content of the image."
        f"Keep the description short and focused, without excessive details. "
        f"Use plain language and make the description suitable for search engines."
    )

    description = extract_image_description(input)

    #print(description)

    if description != None:
        prompt = (
            f"Generate a short description for the image titled '{os.path.basename(input)}'. "
            f"Focus on the main content of the image. Description from metadata: {description}. "
            f"Keep the description short and focused, without excessive details. "
            f"Use plain language and make the description suitable for search engines."
        )
    image = Image.open(input)

    #print(processor.tokenizer.eos_token_id)

    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]

    full_string_output = list_of_faces + "\n" + decoded_text
    return full_string_output


input = "Bilder/191-Preis_fuer_exzellente_Lehre-Gruppe-scaled.jpg"
reference_folder = "person_images"

print(os.path.basename(input))
output = image_to_text_description(input, reference_folder)
print(output)