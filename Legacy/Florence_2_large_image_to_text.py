import requests
from PIL import Image
import os
from pathlib import Path
from extract_metadata import *
from image_to_text_face_recognition import *
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from CLIP_Classification import *

def get_files(path):

    result = []
    path = Path(path)
    
    if path.is_file():
        result.append(str(path))
    elif path.is_dir():
        for item in path.iterdir():
            if item.is_file():
                result.append(str(item))

    return result

def preload_faces(reference_folder):
    reference_faces = load_reference_faces(reference_folder)
    return reference_faces

def custom_split(text_description):
    abbreviations = ["Prof.", "Dr."]
    split_index = -1

    for i in range(len(text_description) - 1):
        if (text_description[i] == '.' and 
            (text_description[i + 1] == ' ' or text_description[i + 1] == '\n')):
            
            if text_description[max(0, i - 4):i + 1] not in abbreviations:
                split_index = i
                break

    # Return the split text
    if split_index != -1:
        return text_description[:split_index + 1].strip()
    return text_description.strip()

def prompt_selector(image_class, faces, image_file):

    print(image_class)
    print("#" * 100)
    print(faces)

    prompt = "<DETAILED_CAPTION>"

    # match image_class:
    #     case "Stock":
    #         prompt = (f"This is a stock image. Generate a one sentence description")

    #     case "Infographic":
    #         prompt = (f"This image is titled: {os.path.basename(image_file)}. This is an infographic. Describe this image in 10-20 words.")

    #     case "Portrait":
    #         prompt = (f"This image is titled: {os.path.basename(image_file)}. This is a portrait. Describe the Person in this image objectively. Do not give your opinion.")

    #     case "Event":
    #         prompt = (f"This image is titled: {os.path.basename(image_file)}. Do not give your opinion. This is an image of an event. Describe this image in 10-30 words.")

    #     case "Logo":
    #             prompt = (f"This image is titled: {os.path.basename(image_file)}. This is a logo. Describe this logo in 5-10 words.")

    return prompt

def image_to_text_description(image_file, preloaded_reference_faces):

    list_of_faces = generate_string_of_faces(image_file, preloaded_reference_faces)

    image_class = classify_image(image_file) #Stock Infographic Portrait Event Logo

    if image_class == "Stock":
        if list_of_faces != "":
            image_class = "Event"
        else:
            return "Stock image"

    prompt = prompt_selector(image_class, list_of_faces, image_file)

    model_id = "microsoft/Florence-2-large"

    print(prompt)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    raw_image = Image.open(image_file)
    inputs = processor(text=prompt, images=raw_image, return_tensors='pt').to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
    )

    text_description = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Parse the Florence-2 specific output if necessary
    # text_description = processor.post_process_generation(
    #     text_description, 
    #     task="<OD>", 
    #     image_size=(raw_image.width, raw_image.height)
    # )

    #text_description = custom_split(text_description)

    full_string_output = image_class + "\n" + list_of_faces + "\n" + text_description
    return full_string_output

reference_folder = "person_images"
preloaded_faces = preload_faces(reference_folder)

input_image_file = "TestBilder1"

array_of_images = get_files(input_image_file)

output_file = "output_prompts.txt"
with open(output_file, "w") as file:
    for image in array_of_images:
        print(os.path.basename(image))
        output = image_to_text_description(image, preloaded_faces)
        
        # Write the image name and generated prompt to the output file
        file.write(f"Image Name: {os.path.basename(image)}\n")
        file.write(f"Generated Description:\n{output}\n")
        file.write("=" * 50 + "\n")  # Add a separator for readability
