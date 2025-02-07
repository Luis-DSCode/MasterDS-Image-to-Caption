import face_recognition
import os
from PIL import Image
import numpy as np

#Load reference images and extract face encodings
def load_reference_faces(reference_images_folder):
    reference_faces = {}
    
    for filename in os.listdir(reference_images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(reference_images_folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                reference_faces[filename] = face_encodings[0]
                
    return reference_faces

#Check if the face in the image matches
def check_for_matching_faces(image_path, reference_faces):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if not face_encodings:
        return

    matches = []
    for i, face_encoding in enumerate(face_encodings):

        for ref_name, ref_encoding in reference_faces.items():
            match = face_recognition.compare_faces([ref_encoding], face_encoding, tolerance=0.5)
            if match[0]:
                name_without_extension = os.path.splitext(ref_name)[0]
                matches.append(name_without_extension)
    
    return matches if matches else ""

# Step 3: Update the description with face matching info
def generate_string_of_faces(image_path, reference_faces):
    
    # Get face match results
    matching_faces = check_for_matching_faces(image_path, reference_faces)
    
    # Generate the description based on whether faces match
    if isinstance(matching_faces, list):
        matching_faces = list(set(matching_faces))
    
    # Generate the description based on whether faces match
    if matching_faces:
        description = f"{', '.join(matching_faces)} can be seen in this image."
    else:
        description = ""
        
    return description

# Example usage
#reference_images_folder = "person_images"
#reference_faces = load_reference_faces(reference_images_folder)
#image_to_check = "Bilder/1712149147090.jpg"
#description = generate_string_of_faces(image_to_check, reference_faces)

#print(description)