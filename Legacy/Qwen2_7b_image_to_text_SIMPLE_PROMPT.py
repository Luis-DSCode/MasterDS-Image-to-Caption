import requests
from PIL import Image
import os
from pathlib import Path
from extract_metadata import *
from image_to_text_face_recognition import *
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration
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

def resize_image(image_file, max_width=1000):
    image = Image.open(image_file).convert("RGB")
    width, height = image.size
    if width > max_width:

        new_height = int((max_width / width) * height)
        image = image.resize((max_width, new_height), Image.LANCZOS)
    return image

def prompt_selector(image_class, faces, image_file):
    match image_class:
        case "Stock":
            prompt = f"This is a stock image. Generate a one-sentence description."
        case "Infographic":
            prompt = f"This is an infographic titled '{os.path.basename(image_file)}'. Generate a short caption."
        case "Portrait":
            prompt = f"This is a portrait titled '{os.path.basename(image_file)}'. Generate a short caption."
        case "Event":
            prompt = f"This is an image of an event titled '{os.path.basename(image_file)}'. Generate a short caption."
        case "Logo":
            prompt = f"This is a logo titled '{os.path.basename(image_file)}'. Generate a short caption."
    return prompt

def image_to_text_description(image_file, preloaded_reference_faces):
    list_of_faces = generate_string_of_faces(image_file, preloaded_reference_faces)
    image_class = classify_image(image_file) #Stock, Infographic, Portrait, Event, Logo

    if image_class == "Stock" and list_of_faces:
        image_class = "Event"
    elif image_class == "Stock":
        return "Stock image"

    prompt = "Generate a short caption"
    

    raw_image = resize_image(image_file)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate description
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text_description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Post-process the output
    #text_description = custom_split(text_description)
    full_string_output = f"{list_of_faces}\n{text_description}"
    torch.cuda.empty_cache()
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