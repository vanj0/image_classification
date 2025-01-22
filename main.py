import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from Model import Model
from ImageLoader import ImageLoader

class ImageUploaderApp:
    def __init__(self, window):
        self.root = window
        self.root.title("Image Classifier")
        self.root.geometry("400x500")

        # Load and train the model
        self.model = Model(model_file='trained_model.keras')
        self.train_model()


        self.label = tk.Label(window, text="Upload an image to predict the digit", font=("Helvetica", 14))
        self.label.pack(pady=20)

        self.upload_button = tk.Button(window, text="Upload Image", command=self.upload_image, font=("Helvetica", 12))
        self.upload_button.pack(pady=10)

        self.result_label = tk.Label(window, text="Prediction: ", font=("Helvetica", 12))
        self.result_label.pack(pady=20)

        self.image_label = tk.Label(window)
        self.image_label.pack(pady=20)

    def train_model(self):

        path = r'D:\New folder\PyCharm 2023.2\Image_Classification\image_classification\images'

        image_loader = ImageLoader(split_size=0.5)

        # loading samples
        train_samples, train_labels, val_samples, val_labels = image_loader.load(path)

        # training the model
        self.model.train(train_samples, train_labels, learning_rate=0.01, preference=0.02)
        print(f"Model trained on {len(train_samples)} samples.")

    def upload_image(self):

        file_path = filedialog.askopenfilename(title="Select an Image",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            image = Image.open(file_path)
            image_resized = image.resize((200, 200))

            image_tk = ImageTk.PhotoImage(image_resized)

            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk

            predicted_label, predicted_probability = self.model.predict(image)

            self.result_label.config(text=f"Prediction: {predicted_label} (Probability: {predicted_probability:.3f})")
        else:
            messagebox.showerror("Error", "No file selected.")




window = tk.Tk()
app = ImageUploaderApp(window)
window.mainloop()
