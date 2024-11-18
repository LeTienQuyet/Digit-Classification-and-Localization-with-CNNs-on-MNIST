from tkinter import colorchooser, messagebox
from PIL import ImageGrab, Image
from predict import load_model
from train import make_transform

import torch
import tkinter as tk
import matplotlib.pyplot as plt

class PaintApp:
    def __init__(self, root, bg_color="black", text_color="white", width=420, height=420, size=3):
        self.root = root
        self.bg_color = bg_color
        self.text_color = text_color
        self.size = size
        self.width = width
        self.height = height

        self.model = load_model(out_channels=10, path_to_checkpoint="../checkpoint")
        self.scaleX = self.width / 70
        self.scaleY = self.height / 70

        self.root.title("Predict digit with Tkinter")

        self.canvas = tk.Canvas(self.root, bg=bg_color, width= self.width, height=self.height)
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        btn_clear.pack(side=tk.LEFT, expand=True)

        btn_save = tk.Button(self.root, text="Predict", command=self.make_predict)
        btn_save.pack(side=tk.LEFT, expand=True)

    def paint(self, event):
        x, y = event.x, event.y
        r = self.size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=self.text_color, outline=self.text_color)

    def reset(self, event):
        pass

    def clear_canvas(self):
        self.canvas.delete("all")

    def make_predict(self):
        xmin = self.root.winfo_rootx() + self.canvas.winfo_x()
        ymin = self.root.winfo_rooty() + self.canvas.winfo_y()
        xmax = xmin + self.canvas.winfo_width()
        ymax = ymin + self.canvas.winfo_height()

        image = ImageGrab.grab((xmin, ymin, xmax, ymax)).convert("L")
        img = make_transform()(image)
        img = img.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            class_logits, bbox_out = self.model(img)
            class_out = torch.argmax(class_logits, dim=1)
            x, y, w, h = bbox_out[0].cpu().numpy()
            self.canvas.create_rectangle(self.scaleX*x, self.scaleY*y, self.scaleX*(x+w), self.scaleY*(y+h), fill=None, outline="red")
            self.canvas.create_text(self.scaleX*(x+w/2), self.scaleY*(y)-12, text=f"{class_out.item()}", font=("Arial, 15"), fill="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()