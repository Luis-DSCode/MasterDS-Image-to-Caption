import requests
from PIL import Image
import os
from pathlib import Path
from extract_metadata import *
from image_to_text_face_recognition import *
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from CLIP_Classification import *
import tempfile
import imghdr

def get_files(path):
    result = []
    
    if path.startswith("http://") or path.startswith("https://"):
        try:
            response = requests.get(path, stream=True)
            response.raise_for_status() 
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") 
            with open(temp_file.name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            result.append(temp_file.name)
        except Exception as e:
            print(f"Error downloading image from {path}: {e}")
    
    else:
        path = Path(path)
        if path.is_file():
            if imghdr.what(path):
                result.append(str(path))
            else:
                print(f"Skipped non-image file: {path}")
        elif path.is_dir():
            for item in path.iterdir():
                if item.is_file() and imghdr.what(item):
                    result.append(str(item))
                elif item.is_file():
                    print(f"Skipped non-image file: {item}")

    return result


def preload_faces(reference_folder):
    reference_faces = load_reference_faces(reference_folder)
    return reference_faces

def prompt_selector(image_class, faces, image_file):

    # print("#" * 100)
    # print(faces)

    match image_class:
        case "Stock":
            prompt = f"This is a stock image. Generate a one-sentence description."
        case "Infographic":
            prompt = f"This is an infographic. The image file is titled '{os.path.basename(image_file)}'. Generate a short caption that describes the key information presented in the image."
        case "Portrait":
            prompt = f"This is a portrait. {faces} The image file is titled titled '{os.path.basename(image_file)}'. Generate a caption."
        case "Event":
            prompt = f"This is an image of an event. The image file is titled '{os.path.basename(image_file)}'. Generate a caption."
        case "Logo":
            prompt = f"This is a logo. The image file is titled '{os.path.basename(image_file)}'. Generate a short caption."
    return prompt


def image_to_text_description(image_file, preloaded_reference_faces):

    list_of_faces = generate_string_of_faces(image_file, preloaded_reference_faces)
    image_class = classify_image(image_file)  # Stock, Infographic, Portrait, Event, Logo
    #print(image_class)

    if image_class == "Stock":
        if list_of_faces != "":
            image_class = "Event"
        else:
            return "Stock image"

    #prompt = prompt_selector(image_class, list_of_faces, image_file)
    prompt = "<|image_1|>\nDescribe this image objectively."

    model_id = "microsoft/Phi-3.5-vision-instruct"

    #print(prompt)

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

    raw_image = Image.open(image_file)
    placeholder = "<|image|>"

    messages = [{"role": "user", "content": placeholder + prompt}]
    chat_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(chat_prompt, [raw_image], return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args,
    )

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    text_description = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    full_string_output = list_of_faces + "\n" + text_description
    return full_string_output


def generate_image_captions(input_path, reference_folder):

    preloaded_faces = preload_faces(reference_folder)

    array_of_images = get_files(input_path)

    captions = []

    for image in array_of_images:
        #print(f"Processing: {os.path.basename(image)}")
        output = image_to_text_description(image, preloaded_faces)

        captions.append({
            "image_name": os.path.basename(image),
            "caption": output
        })

    return captions

#Example use
#generate_image_captions(input_image, face_referance_folder)
#input_image can be a directory, image file or URL
#face_referance_folder has to be a folder with all faces to be recognized. Filename will be used as name of the person when detected
output = generate_image_captions("TestBilder1", "person_images")

# Save results to a file
output_file = "output_captions.txt"
with open(output_file, "w") as file:
    for caption in output:
        file.write(f"Image Name: {caption['image_name']}\n")
        file.write(f"Generated Description:\n{caption['caption']}\n")
        file.write("=" * 50 + "\n")  # Add a separator for readability

# Print the results
for caption in output:
    print(f"Image Name: {caption['image_name']}")
    print(f"Generated Description: {caption['caption']}")
    print("=" * 50)