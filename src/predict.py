from train import CustomNeuralNetwork, make_transform
from PIL import Image
import os
import torch

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

def main(path_to_checkpoint, path_to_img):
    model = load_model(path_to_checkpoint)
    class_out, bbox_out = predict(model, path_to_img)
    print(f"Number = {class_out.item()}, Bounding box = {bbox_out[0].tolist()}")

if __name__ == "__main__":
    path_to_checkpoint = "../checkpoint"
    path_to_img = "../enlarged-mnist-data-with-bounding-boxes/mnist_img_with_bb/0/0_1.png"
    main(path_to_checkpoint, path_to_img)