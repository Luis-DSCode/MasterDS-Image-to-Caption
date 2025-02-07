from PIL import Image
import exifread

def extract_image_description(filepath):
    image = Image.open(filepath)
    image_format = image.format

    if image_format == "JPEG":
        return extract_jpeg_description(filepath)

    elif image_format == "PNG":
        return extract_png_description(filepath)
    
    else:
        return

def extract_jpeg_description(filepath):
    with open(filepath, 'rb') as file:
        tags = exifread.process_file(file)
        
        # Versuch, die "ImageDescription" zu finden
        description = tags.get("Image ImageDescription")
        return description

def extract_png_description(filepath):
    with Image.open(filepath) as img:
        metadata = img.info

        description = metadata.get("Description")

        return description

#image_description = extract_image_description("Bilder/iStock-1153242630-e1725285448500.jpg")
#print(image_description)