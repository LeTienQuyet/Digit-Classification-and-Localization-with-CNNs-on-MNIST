from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

def read_file_csv(path_to_csv_file):
    df = pd.read_csv(path_to_csv_file)
    train_data_df, test_data_df = train_test_split(df, test_size=0.2, random_state=42)
    train_data_df, val_data_df = train_test_split(train_data_df, test_size=0.25, random_state=42)
    return train_data_df, val_data_df, test_data_df

class CustomDataset(Dataset):
    def __init__(self, path_to_dir, data, transform=None):
        super(CustomDataset, self).__init__()
        self.path_to_dir = path_to_dir
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, img_name, label, h, w, x, _, y, _ = self.data.iloc[idx]
        img_name = os.path.join(self.path_to_dir, str(label), img_name)
        img = Image.open(img_name)
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, bbox

def make_transform():
    transform = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.ToTensor()
    ])
    return transform

def prepare_data(path_to_dir, path_to_csv_file, transform, batch_size=256):
    train_data, val_data, test_data = read_file_csv(path_to_csv_file)

    train_dataset = CustomDataset(path_to_dir, train_data, transform)
    val_dataset = CustomDataset(path_to_dir, val_data, transform)
    test_dataset = CustomDataset(path_to_dir, test_data, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

class CustomNeuralNetwork(nn.Module):
    def __init__(self, out_channels):
        super(CustomNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.batch_norm1 = nn.BatchNorm2d(num_features=16)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)

        self.classification_head = nn.Linear(in_features=128, out_features=out_channels)
        self.bbox_head = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.batch_norm1(self.max_pooling1(x))
        x = F.relu(x)
        x = self.conv4(self.conv3(x))
        x = self.batch_norm2(self.max_pooling2(x))
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        class_logits = self.classification_head(x)
        bbox_coords = self.bbox_head(x)
        return class_logits, bbox_coords

def val_model(model, val_dataloader, classify_loss_fn, regression_loss_fn, optimizer, device):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for (img, label_target, bbox_target) in val_dataloader:
            img, label_target, bbox_target = img.to(device), label_target.to(device), bbox_target.to(device)

            label_out, bbox_out = model(img)
            classify_loss = classify_loss_fn(label_out, label_target)
            bbox_loss = regression_loss_fn(bbox_out, bbox_target)
            loss = classify_loss + bbox_loss
            total_val_loss += loss.item()
    return total_val_loss

def train_model(num_epochs, model, train_dataloader, val_dataloader, classify_loss_fn, regression_loss_fn, optimizer,
                device, path_to_save):
    model.to(device)
    train_losses = []
    val_losses = []
    min_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for (img, label_target, bbox_target) in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch",
                                                     colour="RED"):
            img, label_target, bbox_target = img.to(device), label_target.to(device), bbox_target.to(device)

            optimizer.zero_grad()

            label_out, bbox_out = model(img)
            classify_loss = classify_loss_fn(label_out, label_target)
            bbox_loss = regression_loss_fn(bbox_out, bbox_target)
            loss = classify_loss + bbox_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        train_losses.append(total_train_loss)

        total_val_loss = val_model(model, val_dataloader, classify_loss_fn, regression_loss_fn, optimizer, device)
        val_losses.append(total_val_loss)

        if total_val_loss < min_loss:
            min_loss = total_val_loss
            torch.save(model.state_dict(), os.path.join(path_to_save, "best_model.pt"))

        torch.save(model.state_dict(), os.path.join(path_to_save, "last_model.pt"))
        print(f"train_loss = {total_train_loss:.5f}, val_loss = {total_val_loss:.5f}\n")
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def main(num_epochs, lr, batch_size):
    train_dataloader, val_dataloader, test_dataloader = prepare_data(
        path_to_dir="../enlarged-mnist-data-with-bounding-boxes/mnist_img_with_bb",
        path_to_csv_file="../enlarged-mnist-data-with-bounding-boxes/mnist_img_enlarged_bb.csv",
        transform=make_transform(),
        batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Epochs = {num_epochs}, Learning rate = {lr}, Batch = {batch_size}, Device: {device}")

    out_channels = 10
    model = CustomNeuralNetwork(out_channels)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    classify_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.MSELoss()

    train_model(
        num_epochs=num_epochs, model=model, device=device,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        classify_loss_fn=classify_loss_fn, regression_loss_fn=regression_loss_fn,
        optimizer=optimizer, path_to_save="../checkpoint/"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")
    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=30)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
    args = parser.parse_args()

    main(num_epochs=args.epoch, lr=args.lr, batch_size=args.batch_size)



