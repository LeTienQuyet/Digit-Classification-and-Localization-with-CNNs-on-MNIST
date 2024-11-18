from tkinter import colorchooser, messagebox
from PIL import ImageGrab

import tkinter as tk

class PaintApp:
    def __init__(self, root, bg_color="black", text_color="white", width=400, height=400, size=5):
        self.root = root
        self.bg_color = bg_color
        self.text_color = text_color
        self.size = size
        self.width = width
        self.height = height

        self.root.title("Predict digit with Tkinter")

        self.canvas = tk.Canvas(self.root, bg=bg_color, width= self.width, height=self.height)
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        btn_clear.pack(side=tk.LEFT, expand=True)

        btn_save = tk.Button(self.root, text="Predict", command=self.save_canvas)
        btn_save.pack(side=tk.LEFT, expand=True)

    def paint(self, event):
        x, y = event.x, event.y
        r = self.size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=self.text_color, outline=self.text_color)

    def reset(self, event):
        pass

    def clear_canvas(self):
        self.canvas.delete("all")

    def save_canvas(self):
        xmin = self.root.winfo_rootx() + self.canvas.winfo_x()
        ymin = self.root.winfo_rooty() + self.canvas.winfo_y()
        xmax = xmin + self.canvas.winfo_width()
        ymax = ymin + self.canvas.winfo_height()

        image = ImageGrab.grab((xmin, ymin, xmax, ymax))
        # x_out, y_out, w_out, h_out = 20, 20, 100, 100
        # self.canvas.create_rectangle(x_out, y_out, x_out+w_out, y_out+h_out, fill=None, outline="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()