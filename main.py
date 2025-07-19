import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label, messagebox
from PIL import Image, ImageTk
import os

LEFT_CROP = 350  # Only process this many pixels from the left
CROWN_TEMPLATE = "crown.png"
CROWN_THRESHOLD = 0.85  # Template match threshold

def remove_crowns_and_numbers(image_path, crown_template_path, left_crop=LEFT_CROP):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image.")
    h, w = img.shape[:2]
    left_img = img[:, :left_crop].copy()
    right_img = img[:, left_crop:].copy()

    crown = cv2.imread(crown_template_path)
    if crown is None:
        raise Exception("Could not read crown template.")
    crown_gray = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY)
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(left_gray, crown_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= CROWN_THRESHOLD)

    crown_h, crown_w = crown.shape[:2]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(left_img, pt, (pt[0] + crown_w, pt[1] + crown_h), (255,255,255), -1)
        num_top = pt[1] + crown_h + 5
        num_bottom = num_top + 35  # Adjust if needed
        cv2.rectangle(left_img, (pt[0], num_top), (pt[0] + crown_w, num_bottom), (255,255,255), -1)

    result = np.hstack([left_img, right_img])
    return result

def process_image_gui():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    try:
        result = remove_crowns_and_numbers(file_path, CROWN_TEMPLATE)
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, result)
            messagebox.showinfo("Success", f"Saved cleaned image to:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def main():
    root = Tk()
    root.title("Leaderboard Cleaner")
    root.geometry("350x150")
    Label(root, text="Leaderboard Image Cleaner", font=("Arial", 16)).pack(pady=10)
    Button(root, text="Select and Clean Image", command=process_image_gui, width=25, height=2).pack(pady=20)
    Label(root, text="Removes crowns and numbers on left side.\nSupports PNG/JPG.", font=("Arial", 10)).pack()
    root.mainloop()

if __name__ == "__main__":
    main()