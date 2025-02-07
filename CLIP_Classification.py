import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_image(image_path):
    labels = [
        "stock photograph", "digital graph", "digital chart", "logo", 
        "portrait photograph", "event photograph", "infographic", "landscape"
    ]

    image = Image.open(image_path)

    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    label_prob_pairs = [(labels[idx], prob.item()) for idx, prob in enumerate(probs[0])]
    label_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    label_stock_photograph = probs[0][labels.index("stock photograph")].item()
    label_digital_graph = probs[0][labels.index("digital graph")].item()
    label_digital_chart = probs[0][labels.index("digital chart")].item()
    label_logo = probs[0][labels.index("logo")].item()
    label_portrait_photograph = probs[0][labels.index("portrait photograph")].item()
    label_event_photograph = probs[0][labels.index("event photograph")].item()
    label_infographic = probs[0][labels.index("infographic")].item()
    label_landscape = probs[0][labels.index("landscape")].item()

    is_stock = label_stock_photograph + 1.25 * label_landscape
    is_infographic = label_digital_graph + label_digital_chart + label_infographic
    is_portrait = label_portrait_photograph
    is_event = label_event_photograph + 0.25 * label_stock_photograph
    is_logo = label_logo

    highest_category = max(
    ("Stock", is_stock),
    ("Infographic", is_infographic),
    ("Portrait", is_portrait),
    ("Event", is_event),
    ("Logo", is_logo),
    key=lambda x: x[1]
    )
    return highest_category[0]



# image_path = "TestBilder1/iStock-1153242630-e1725285448500.jpg"
# classify_image(image_path)

# (label_stock_photograph, label_digital_graph, label_digital_chart, label_logo, 
#  label_portrait_photograph, label_event_photograph, label_infographic, label_landscape) = classify_image(image_path)

    # print("Stock Photograph Probability:", label_stock_photograph)
    # print("Digital Graph Probability:", label_digital_graph)
    # print("Digital Chart Probability:", label_digital_chart)
    # print("Logo Probability:", label_logo)
    # print("Portrait Photograph Probability:", label_portrait_photograph)
    # print("Event Photograph Probability:", label_event_photograph)
    # print("Infographic Probability:", label_infographic)
    # print("Landscape Probability:", label_landscape)

    # print(highest_category)
