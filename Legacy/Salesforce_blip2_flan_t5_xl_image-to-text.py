from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os
from extract_metadata import *

# Load the BLIP-2 model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip2-flan-t5-xl"  # Use "Salesforce/blip2-opt-2.7b" if you want more power and have the resources
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)

# Define the image path
input_image_path = "Bilder/iStock-1153242630-e1725285448500.jpg"
image = Image.open(input_image_path)

# Extract filename and pre-existing description if available
filename = os.path.basename(input_image_path)
description = extract_image_description(input_image_path)

print(description)

# Construct a more explicit, search-optimized prompt
if description:
    prompt = (
        f"This is an image of a stock photo. The file is named '{filename}', and the description of the image includes: '{description}'. "
        "Please describe the content of the image in detail, identifying the people, the environment, and any activities taking place. "
        "Describe the individuals in the image, including their physical characteristics, clothing, and actions. "
        "Also describe the setting, the overall mood of the image, and the lighting. Is this a typical university or professional environment? "
        "Finally, identify the atmosphere of the image, focusing on concentration, teamwork, or focus, and any other elements that stand out."
    )
else:
    prompt = (
        f"This is an image of a stock photo. The file is named '{filename}'. Please describe the content of the image in detail, "
        "identifying the people, the environment, and any activities taking place. Describe the individuals in the image, "
        "including their physical characteristics, clothing, and actions. Also describe the setting, the overall mood of the image, "
        "and the lighting. Is this a typical university or professional environment? Finally, identify the atmosphere of the image, "
        "focusing on concentration, teamwork, or focus, and any other elements that stand out."
    )
    
# Prepare inputs for the model with expanded image tokens
# First, process the image and text
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

# Generate output with BLIP-2
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256)

# Decode and print the generated description
generated_description = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"Filename: {filename}")
print(f"Generated Description: {generated_description}")