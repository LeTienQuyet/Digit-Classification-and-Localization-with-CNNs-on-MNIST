from train import CustomNeuralNetwork, make_transform
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch
import argparse
import time

def load_model(path_to_checkpoint, out_channels=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = CustomNeuralNetwork(out_channels=out_channels)
    best_model.load_state_dict(torch.load(os.path.join(path_to_checkpoint, "best_model.pt"), map_location=device, weights_only=True))
    return best_model

def predict(model, path_to_img):
    img = Image.open(path_to_img)
    img_rgb = img.convert("L")
    img_rgb = make_transform()(img_rgb).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        class_logits, bbox_out = model(img_rgb)
        class_out = torch.argmax(class_logits, dim=1)

    img_name, img_format = path_to_img.rsplit('.', 1)
    output_name = f"{img_name}_predicted.{img_format}"
    x, y, w, h = bbox_out[0].tolist()

    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap="gray")
    img_predicted = patches.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
    ax.text(x + w / 3, y - 0.5, f"Predicted: {class_out.item()}", color='white', fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.add_patch(img_predicted)
    plt.axis('off')
    plt.savefig(os.path.join("../image", output_name), bbox_inches='tight', pad_inches=0)
    plt.close()

    return class_out, bbox_out

def main(path_to_checkpoint, img_name):
    model = load_model(path_to_checkpoint)
    path_to_img = os.path.join("../image", img_name)
    class_out, bbox_out = predict(model, path_to_img)
    return class_out, bbox_out

if __name__ == "__main__":
    path_to_checkpoint = "../checkpoint"

    parser = argparse.ArgumentParser(description="Image for predict")
    parser.add_argument("--image", type=str, help="Name of image save in `image`", default="example.png")
    args = parser.parse_args()

    start_time = time.time()
    class_out, bbox_out = main(path_to_checkpoint, img_name=args.image)
    end_time = time.time()
    print(f"Predicted time : {end_time - start_time:.4f} ms")