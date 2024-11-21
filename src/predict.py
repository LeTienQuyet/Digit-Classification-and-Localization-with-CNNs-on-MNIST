from train import CustomNeuralNetwork, make_transform
from PIL import Image

import os
import torch
import argparse

def load_model(path_to_checkpoint, out_channels=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = CustomNeuralNetwork(out_channels=out_channels)
    best_model.load_state_dict(torch.load(os.path.join(path_to_checkpoint, "best_model.pt"), map_location=device, weights_only=True))
    return best_model

def predict(model, path_to_img):
    img = Image.open(path_to_img).convert("L")
    img = make_transform()(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        class_logits, bbox_out = model(img)
        class_out = torch.argmax(class_logits, dim=1)
    return class_out, bbox_out

def main(path_to_checkpoint, img_name):
    model = load_model(path_to_checkpoint)
    path_to_img = os.path.join("../image", img_name)
    class_out, bbox_out = predict(model, path_to_img)
    return class_out, bbox_out

if __name__ == "__main__":
    path_to_checkpoint = "../checkpoint"

    parser = argparse.ArgumentParser(description="Image for predict")
    parser.add_argument("image", type=str, help="Name of image save in `image`")
    args = parser.parse_args()

    class_out, bbox_out = main(path_to_checkpoint, img_name=args.image)
    print(f"Number = {class_out.item()}, Bounding box = {bbox_out[0].tolist()}")