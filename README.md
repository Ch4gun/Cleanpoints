# Cleanpoints

A simple Python GUI tool to clean up leaderboard images for OCR (Tesseract).

## Features

- Removes crown icons and "might" numbers under nicknames on the left side of leaderboard images.
- Preserves points on the right side.
- Supports `.png` and `.jpg` images.
- Easy-to-use Windows GUI.

## Requirements

- Python 3.7+
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) (for later OCR, not required for cleaning)
- Python packages:
  ```
  pip install opencv-python pillow numpy
  ```

## Usage

1. Place `crown.png` (the crown template) in the same folder as `main.py`.
2. Run the app:
   ```
   python main.py
   ```
3. Click "Select and Clean Image", choose your leaderboard image, and save the cleaned result.

## Customization

- Adjust `LEFT_CROP` in `main.py` if your leaderboard layout changes.
- Adjust the rectangle size for number removal if needed.

## License

MIT