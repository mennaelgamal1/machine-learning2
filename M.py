import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib

svm_model = joblib.load('svm_fruit_classifier.pkl')
scaler = joblib.load('svm_scaler.pkl')

class_labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
img_size = (100, 100)

# ---------- Predict Fruit ----------
def predict_fruit(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0  # normalize to [0,1]
    img_flatten = img_array.flatten().reshape(1, -1)

    img_scaled = scaler.transform(img_flatten)
    predicted_index = svm_model.predict(img_scaled)[0]

    return class_labels[predicted_index], img_array

# ---------- GUI Setup ----------
root = tk.Tk()
root.title("Fruit Classifier")
root.geometry("550x600")
root.configure(bg="#f0fff0")

header = tk.Label(root, text="Fruit Classification", font=("Helvetica", 18, "bold"), bg="#f0fff0", fg="#2e7d32")
header.pack(pady=10)

image_label = tk.Label(root, bg="#f0fff0")
image_label.pack(pady=10)

result_label = tk.Label(root, text="Upload an image to classify the fruit", font=("Arial", 14), bg="#f0fff0", fg="#1b5e20")
result_label.pack(pady=10)

# ---------- Show Prediction Result ----------
def show_prediction(img_array, label):
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='RGB')
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    result_label.config(text=f"Prediction: {label}", fg="#1b5e20")

# ---------- Upload Image Button ----------
def upload_image():
    file_path = filedialog.askopenfilename(title="Select a fruit image")
    if not file_path or not os.path.isfile(file_path):
        result_label.config(text="Invalid file selected.", fg="red")
        return

    label, img_array = predict_fruit(file_path)
    show_prediction(img_array, label)

# ---------- Buttons ----------
upload_btn = tk.Button(root, text="Upload Image", font=("Arial", 12), command=upload_image,
                       bg="#4caf50", fg="white", padx=10, pady=5)
upload_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit,
                     bg="#f44336", fg="white", padx=10, pady=5)
exit_btn.pack(pady=5)

root.mainloop()
