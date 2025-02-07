import requests
from PIL import Image
import os
from pathlib import Path
from extract_metadata import *
from image_to_text_face_recognition import *
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from CLIP_Classification import *
import tempfile

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image content
        return True
    except (IOError, SyntaxError):
        return False

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
            if is_valid_image(temp_file.name):
                result.append(temp_file.name)
            else:
                print(f"Downloaded file is not a valid image: {temp_file.name}")
        except Exception as e:
            print(f"Error downloading image from {path}: {e}")
    
    else:
        path = Path(path)
        if path.is_file():
            if is_valid_image(path):
                result.append(str(path))
            else:
                print(f"Skipped invalid image file: {path}")
        elif path.is_dir():
            for item in path.iterdir():
                if item.is_file() and is_valid_image(item): 
                    result.append(str(item))
                elif item.is_file():
                    print(f"Skipped non-image file: {item}")

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
        #+ "."
    return text_description.strip()
    #+ "."

def prompt_selector(image_class, faces, image_file):

    # print(image_class)
    # print("#" * 100)
    # print(faces)

    match image_class:
        case "Stock":
            prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"This is a stock image. Generate a one sentence description.\n<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        case "Infographic":
            prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"<image>\nThis image is titled: {os.path.basename(image_file)}. This is an infographic. Describe this image in 10-20 words.\n<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        case "Portrait":
            #if faces == "":
            prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"<image>\nThis image is titled: {os.path.basename(image_file)}. This is a portrait. Describe the Person in this image objectively. Do not give your opinion.\n<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
            # else:
            #     prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n"
            #             f"<image>\nThis image is titled: {os.path.basename(image_file)}.  This is a portrait. Describe the Person in this image objectively. {faces} Do not give your opinion.\n<|eot_id|>"
            #             f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        case "Event":
            prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"<image>\nThis image is titled: {os.path.basename(image_file)}. Do not give your opinion. This is an image of an event. Describe this image in 10-30 words.\n<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        case "Logo":
                prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n"
                        f"<image>\nThis image is titled: {os.path.basename(image_file)}. This is a logo. Describe this logo in 5-10 words.\n<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n")

    #print(prompt)
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

    model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"

    print(prompt)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)


    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
      )

    text_description = (processor.decode(output[0][2:], skip_special_tokens=True))

    if "assistant" in text_description:
        text_description = text_description.split("assistant", 1)[-1].strip()

    #if image_class == "Event":
    #text_description = text_description.split(". ")[0].split(".\n")[0].strip() + "."

    text_description = custom_split(text_description)

    #full_string_output = list_of_faces + "\n" + text_description
    full_string_output = text_description
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
output = generate_image_captions("TestBilder3", "person_images")

# Save results to a file
# output_file = "output_captions.txt"
# with open(output_file, "w") as file:
#     for caption in output:
#         file.write(f"Image Name: {caption['image_name']}\n")
#         file.write(f"Generated Description:\n{caption['caption']}\n")
#         file.write("=" * 50 + "\n")  # Add a separator for readability

# Print the results
for caption in output:
    print(f"Image Name: {caption['image_name']}")
    print(f"Generated Description: {caption['caption']}")
    print("=" * 50)

import csv

# Save results to a CSV file
output_csv_file = "TestBilder3_PHI.csv"

# Write to CSV
with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(["Name", "Beschreibung"])
    
    # Write image captions
    for caption in output:
        csv_writer.writerow([caption["image_name"], caption["caption"]])


# reference_folder = "person_images"
# preloaded_faces = preload_faces(reference_folder)

# input_image_file = "TestBilder1"

# array_of_images = get_files(input_image_file)

# for image in array_of_images:
#     print(os.path.basename(image))
#     output = image_to_text_description(image, preloaded_faces)
#     print("#" * 30)
#     print(output)
#     print("#" * 30)

# output_file = "output_captions.txt"
# with open(output_file, "w") as file:
#     for image in array_of_images:
#         print(os.path.basename(image))
#         output = image_to_text_description(image, preloaded_faces)
        
#         # Write the image name and generated prompt to the output file
#         file.write(f"Image Name: {os.path.basename(image)}\n")
#         file.write(f"Generated Description:\n{output}\n")
#         file.write("=" * 50 + "\n")  # Add a separator for readability

#print(os.path.basename(image_file))

#output = image_to_text_description(image_file, preloaded_faces)
#print(output)