"""
Cleanpoints - Leaderboard Image Cleaner and OCR Tool

A Python GUI application for cleaning leaderboard images by removing crown icons
and associated numbers, then extracting data using OCR for further processing.

Author: Ch4gun
License: MIT
"""

import cv2
import numpy as np
import pytesseract
from tkinter import Tk, filedialog, Button, Label, messagebox, ttk, StringVar
import csv
import os
import sys
import re
import json

# Configuration constants
NICKNAME_CROP_RIGHT = 350  # Right edge of nickname column in pixels
POINTS_CROP_LEFT = -200    # Left edge of points column (from right) in pixels
LEFT_CROP = 400           # Number of pixels to process from the left

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Nickname mapping file
MAPPING_FILE = 'nickname_mapping.json'


def load_mapping():
    """
    Load nickname mappings from JSON file.
    
    Returns:
        dict: Dictionary of OCR nickname to correct nickname mappings
    """
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_mapping(mapping):
    """
    Save nickname mappings to JSON file.
    
    Args:
        mapping (dict): Dictionary of OCR nickname to correct nickname mappings
    """
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def ocr_lines(image, psm=6):
    """
    Run Tesseract OCR on image and return a list of lines.
    
    Args:
        image: PIL Image object
        psm (int): Page segmentation mode (default: 6 for uniform text block)
    
    Returns:
        list: List of text lines extracted from the image
    """
    config = f'--psm {psm}'
    text = pytesseract.image_to_string(image, config=config)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines


def remove_crowns_and_numbers(image_path, left_crop=LEFT_CROP):
    """
    Remove crown icons and their associated numbers from leaderboard images.
    
    This function detects orange/yellow colored crown elements using HSV color filtering,
    then removes them along with any numbers detected below them using OCR.
    
    Args:
        image_path (str): Path to the input image
        left_crop (int): Number of pixels to process from the left (default: 400)
    
    Returns:
        numpy.ndarray: Processed image with crowns and numbers removed
    
    Raises:
        Exception: If the image cannot be read
    """
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image.")
    
    h, w = img.shape[:2]
    processed_img = img.copy()

    # Run OCR to detect all numbers in the image
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    try:
        # Configure OCR for number detection only
        ocr_result = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, 
                                              config='--psm 6 -c tessedit_char_whitelist=0123456789,')
        
        # Extract detected numbers with their positions
        detected_numbers = []
        for i in range(len(ocr_result['text'])):
            text = ocr_result['text'][i].strip()
            # Check if this is a number (digits and commas only)
            if text and all(c.isdigit() or c == ',' for c in text) and len(text) > 0:
                left = ocr_result['left'][i]
                top = ocr_result['top'][i]
                width = ocr_result['width'][i]
                height = ocr_result['height'][i]
                detected_numbers.append({
                    'text': text,
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                })
        
        print(f"Detected {len(detected_numbers)} numbers in the image")
        
    except Exception as e:
        print(f"OCR error: {e}")
        detected_numbers = []

    # Convert to HSV for crown color detection
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    
    # Crown color range (orange/yellow)
    lower_orange = np.array([15, 120, 120])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find crown contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} crown contours")
    
    # Process each detected crown
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter small noise
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            print(f"Processing crown at ({x}, {y}) with size {w_box}x{h_box}, area: {area}")
            
            # Skip crowns that are too far left (likely not real crowns)
            if x < 30:
                print(f"Skipping crown at ({x}, {y}) - too far left, likely not a real crown")
                continue
            
            # Remove crown with white rectangle
            cv2.rectangle(processed_img, (x, y), (x + w_box, y + h_box), (255,255,255), -1)
            
            # Find and remove numbers below this crown
            crown_bottom = y + h_box
            numbers_removed = False
            
            for number in detected_numbers:
                # Check if number is below crown and within reasonable distance
                if (number['top'] > crown_bottom and 
                    number['top'] < crown_bottom + 80 and  # Within 80 pixels below crown
                    abs(number['left'] - x) < 300 and  # Within 300 pixels horizontally from crown
                    number['left'] < w * 0.6):  # Not in the far right (points column)
                    
                    print(f"Removing number '{number['text']}' at position ({number['left']}, {number['top']}) below crown at ({x}, {y})")
                    
                    # Remove number with minimal padding
                    remove_left = max(0, number['left'] - 5)
                    remove_right = min(w, number['left'] + number['width'] + 5)
                    remove_top = max(0, number['top'] - 3)
                    remove_bottom = min(h, number['top'] + number['height'] + 3)
                    
                    cv2.rectangle(processed_img, (remove_left, remove_top), 
                                (remove_right, remove_bottom), (255,255,255), -1)
                    numbers_removed = True
            
            if not numbers_removed:
                print(f"No numbers detected by OCR below crown at ({x}, {y}), skipping removal to preserve points")

    return processed_img


def extract_text_from_image(image_path):
    """
    Extract clean text from image using OCR.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        str: Cleaned text extracted from the image
    
    Raises:
        Exception: If text extraction fails
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Could not read image.")
        
        # Convert to PIL for pytesseract
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Run OCR with optimized settings
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(pil_img, config=config)
        
        # Clean up the text
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Remove common OCR artifacts
            line = re.sub(r'[|:;_=°—]+', '', line)
            line = re.sub(r'\s+', ' ', line)
            line = line.strip()
            
            # Skip lines that are just numbers or very short
            if len(line) < 2:
                continue
                
            # Skip lines that are just punctuation or artifacts
            if re.match(r'^[^\w\s,]+$', line):
                continue
                
            # Skip lines that are just single letters or repeated letters
            if re.match(r'^[a-zA-Z]{1,3}$', line) and len(set(line.lower())) <= 2:
                continue
                
            # Skip lines that are just numbers with letters mixed in
            if re.match(r'^\d+\s+\d+', line):
                continue
                
            # Skip lines that are just artifacts
            if re.match(r'^(a|ee+|7|E+)$', line, re.IGNORECASE):
                continue
                
            lines.append(line)
        
        extracted_text = '\n'.join(lines)
        print(f"Extracted {len(lines)} cleaned lines of text from image")
        
        return extracted_text
        
    except Exception as e:
        raise Exception(f"Error extracting text from image: {str(e)}")


def correct_ocr_result(input_file_path, output_file_path):
    """
    Process OCR result and convert to CSV format.
    
    This function reads a text file with OCR results, pairs nicknames with their
    corresponding numbers, applies spelling corrections, and exports to CSV format.
    
    Args:
        input_file_path (str): Path to OCR result text file
        output_file_path (str): Path for output CSV file
    
    Returns:
        int: Number of processed entries
    
    Raises:
        Exception: If processing fails
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        number_pattern = re.compile(r'^[\d,]+$')
        pairs = []
        used_indices = set()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            # Clean up nickname
            nickname = re.sub(r'[|:;_=°—]+', '', line)
            nickname = re.sub(r'\b[a-zA-Z]{1,2}\b', '', nickname).strip()
            nickname = re.sub(r'\s+', ' ', nickname).strip()
            nickname = re.sub(r'\s+[a-zA-Z]{1,2}$', '', nickname).strip()
            
            # If line is a number, skip
            if number_pattern.match(line):
                i += 1
                continue
                
            # Look ahead up to 3 lines for the next unused number
            points = "0"
            for lookahead in range(1, 4):
                if i + lookahead < len(lines):
                    next_line = lines[i + lookahead]
                    if number_pattern.match(next_line) and (i + lookahead) not in used_indices:
                        points = next_line.replace(',', '')
                        used_indices.add(i + lookahead)
                        break
                    # If another nickname is found before a number, stop looking
                    next_nickname = re.sub(r'[|:;_=°—]+', '', next_line)
                    next_nickname = re.sub(r'\b[a-zA-Z]{1,2}\b', '', next_nickname).strip()
                    next_nickname = re.sub(r'\s+', ' ', next_nickname).strip()
                    next_nickname = re.sub(r'\s+[a-zA-Z]{1,2}$', '', next_nickname).strip()
                    if next_nickname and not number_pattern.match(next_nickname) and lookahead > 0:
                        break
                        
            # Format points with commas
            if points != "0":
                try:
                    points = f"{int(points):,}"
                except:
                    points = points
                    
            # Apply spelling corrections
            corrected_nickname = nickname
            if "PANCHOLOCO" in corrected_nickname:
                corrected_nickname = "PANCHO LOCO"
            elif "xelorageniag71" in corrected_nickname:
                corrected_nickname = "xeloragenia971"
            elif "Nights WatchCP" in corrected_nickname:
                corrected_nickname = "Nights Watch CP"
            elif "Unforgivan" in corrected_nickname:
                corrected_nickname = "Unforgiv3n"
            elif "ILy" in corrected_nickname and "II" not in corrected_nickname:
                corrected_nickname = "ILy II"
            elif "PIERRE" in corrected_nickname and "II" not in corrected_nickname:
                corrected_nickname = "PIERRE II"
            elif "DZIKI E" in corrected_nickname:
                corrected_nickname = "DZIKI"
            elif "hostRider" in corrected_nickname:
                corrected_nickname = "GhostRider"
            elif "Peri" in corrected_nickname and "Pericles" not in corrected_nickname:
                corrected_nickname = "Peri II RG"
            elif "Morituri salutant" in corrected_nickname:
                corrected_nickname = "Morituri te Salutant"
            elif "ConantheGreekGi3" in corrected_nickname:
                corrected_nickname = "Conan the Greek G13"
            elif "MiSs FORtUnE" in corrected_nickname:
                corrected_nickname = "Miss FoRtUnE"
            elif "Creeks5Q" in corrected_nickname:
                corrected_nickname = "Creeks59"
            elif "KatharinaDieGrofe" in corrected_nickname:
                corrected_nickname = "Katharina DieGroße"
            elif "UMAR GUJ JAR" in corrected_nickname:
                corrected_nickname = "UMAR GUJJAR"
            elif "Warren" in corrected_nickname and points == "24655":
                corrected_nickname = "Warren"
                points = "24,655"
                
            # Only add if nickname is not empty and not just a number
            if corrected_nickname and not number_pattern.match(corrected_nickname):
                pairs.append([corrected_nickname, points])
            i += 1

        # Write CSV format
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['nickname', 'points'])
            writer.writerows(pairs)
            
        # Also create a text file with the corrected format
        text_output_path = output_file_path.replace('.csv', '_corrected.txt')
        with open(text_output_path, 'w', encoding='utf-8') as textfile:
            for nickname, points in pairs:
                textfile.write(f"{nickname},{points}\n")
                
        return len(pairs)
        
    except Exception as e:
        raise Exception(f"Error processing OCR result: {str(e)}")


def extract_pairs_by_position(image_path):
    """
    Extract nickname-number pairs from images using positional analysis.
    
    This function groups OCR text by vertical position, identifies nicknames vs numbers,
    and matches them based on proximity while filtering out OCR artifacts.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        list: List of [nickname, points] pairs
    """
    import numpy as np
    img = cv2.imread(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    number_pattern = re.compile(r'^\d[\d,]*$')
    
    # Group words into lines by y-coordinate
    lines = []
    current_line = []
    last_y = None
    y_threshold = 15  # pixels
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        if last_y is None or abs(y - last_y) <= y_threshold:
            current_line.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h})
            last_y = y
        else:
            lines.append(current_line)
            current_line = [{'text': text, 'x': x, 'y': y, 'w': w, 'h': h}]
            last_y = y
    if current_line:
        lines.append(current_line)
    
    # Build nicknames and numbers from lines
    nicknames = []
    numbers = []
    artifact_set = set(['|', 'a', 'eee', '°', '=', '__', '-', '>', '7', 'ee', ':', 'Es', 'ry', 'R', 'E)', 'a rc)', 'Zz', 'Zz > 25 E)', 'PIERREI]', 'Kagamand |', '__|', 'Creeks5Q)', 'MiSs', 'Fi', 'RtUnE', 'Il', 'na', 'G', 'Pan', '1', 'Dolly', 'king', 'GUJJAR', 'rc)', 'Es', 'ry', 'R', 'E)', 'a', 'eee', 'ee', '7', '°', '>', 'Es', 'ry', 'R', 'E)', 'a rc)', 'Zz', 'Zz > 25 E)', 'PIERREI]', 'Kagamand |', '__|'])
    
    for line in lines:
        # Sort words in line by x
        line = sorted(line, key=lambda w: w['x'])
        line_text = ' '.join([w['text'] for w in line])
        x, y, w, h = line[0]['x'], line[0]['y'], line[0]['w'], line[0]['h']
        
        # Clean up nickname
        cleaned_line_text = re.sub(r'^[^\w]+|[^\w]+$', '', line_text)
        cleaned_line_text = re.sub(r'\s+', ' ', cleaned_line_text).strip()
        
        # If the line is a number, treat as number, else as nickname
        if number_pattern.match(cleaned_line_text) and len(cleaned_line_text) > 1:
            numbers.append({'text': cleaned_line_text, 'x': x, 'y': y, 'w': w, 'h': h, 'used': False})
        else:
            # Filter out artifacts, very short nicknames, and lines with too few alphabetic characters
            alpha_count = sum(c.isalpha() for c in cleaned_line_text)
            if len(cleaned_line_text) < 3 or cleaned_line_text in artifact_set or alpha_count < 2:
                continue
            nicknames.append({'text': cleaned_line_text, 'x': x, 'y': y, 'w': w, 'h': h})
    
    # Match nicknames with numbers
    pairs = []
    for nick in nicknames:
        best_num = None
        best_dist = float('inf')
        for num in numbers:
            if num['used']:
                continue
            if num['x'] > nick['x'] and 0 <= (num['y'] - nick['y']) < 60:
                dist = (num['x'] - nick['x'])**2 + (num['y'] - nick['y'])**2
                if dist < best_dist:
                    best_dist = dist
                    best_num = num
        points = best_num['text'] if best_num else '0'
        if best_num:
            best_num['used'] = True
        pairs.append([nick['text'], points])
    
    return pairs


def process_image_gui():
    """
    Main GUI application with all user interface elements.
    
    This function creates the main window with buttons for:
    - Image cleaning
    - CSV export and sorting
    - Format and delimiter selection
    """
    # Initialize variables
    clan_members = []
    mapping = {}
    root = Tk()
    root.title("Leaderboard Image Cleaner")
    root.geometry("500x350")
    root.eval('tk::PlaceWindow . center')
    
    # Title
    title_label = Label(root, text="Leaderboard Image Cleaner", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)
    
    # Output format selection
    format_label = Label(root, text="Output Format:", font=("Arial", 10))
    format_label.pack(pady=5)
    format_var = StringVar(value="PNG")
    format_combo = ttk.Combobox(root, textvariable=format_var, values=["PNG", "JPEG"], state="readonly", width=10)
    format_combo.pack(pady=5)
    
    # CSV delimiter selection
    delimiter_label = Label(root, text="CSV Delimiter:", font=("Arial", 10))
    delimiter_label.pack(pady=5)
    delimiter_var = StringVar(value="Tab")
    delimiter_combo = ttk.Combobox(root, textvariable=delimiter_var, values=["Tab", "Comma"], state="readonly", width=8)
    delimiter_combo.pack(pady=5)
    
    def sort_and_export_csv():
        """GUI function for sorting and exporting text data to CSV."""
        try:
            # Select input text file
            input_file = filedialog.askopenfilename(
                title="Select Text File to Sort",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not input_file:
                return
            
            # Select output CSV file
            output_file = filedialog.asksaveasfilename(
                title="Save Sorted CSV As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not output_file:
                return
            
            # Read and process the file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            
            # Remove empty lines and the first line if it's empty
            lines = [line for line in lines if line.strip()]
            if lines and not lines[0].strip():
                lines = lines[1:]
            
            # Pair nicknames with numbers
            pairs = []
            nicknames = []
            numbers = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line is a number (contains only digits and commas)
                if all(c.isdigit() or c == ',' for c in line):
                    numbers.append(line.replace(',', ''))  # Remove commas
                else:
                    nicknames.append(line)
            
            # Pair nicknames with their corresponding numbers
            for i, nickname in enumerate(nicknames):
                if i < len(numbers):
                    points = numbers[i]
                else:
                    points = "0"  # Default if no number available
                pairs.append([nickname, points])
            
            # Sort by points (descending order)
            pairs.sort(key=lambda x: int(x[1]) if x[1].isdigit() else 0, reverse=True)
            
            # Get selected delimiter
            selected_delimiter = delimiter_var.get()
            if selected_delimiter == "Tab":
                selected_delimiter = "\t"
            elif selected_delimiter == "Comma":
                selected_delimiter = ","
            
            # Write to CSV file
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=selected_delimiter)
                writer.writerow(['nickname', 'points'])
                for nickname, points in pairs:
                    writer.writerow([nickname, points])
            
            messagebox.showinfo("Success", f"Sorted data exported to:\n{output_file}\n\nTotal entries: {len(pairs)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    
    def clean_image():
        """GUI function for cleaning leaderboard images."""
        try:
            # Select input image
            file_path = filedialog.askopenfilename(
                title="Select Leaderboard Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )
            if not file_path:
                return
            
            # Get selected format
            selected_format = format_var.get()
            if selected_format == "PNG":
                defaultextension = ".png"
                filetypes = [("PNG files", "*.png"), ("JPEG files", "*.jpg")]
            else:  # JPEG
                defaultextension = ".jpg"
                filetypes = [("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            
            # Select output location
            save_path = filedialog.asksaveasfilename(
                title="Save Cleaned Image As",
                defaultextension=defaultextension,
                filetypes=filetypes
            )
            if not save_path:
                return
            
            # Process image
            result = remove_crowns_and_numbers(file_path)
            cv2.imwrite(save_path, result)
            
            messagebox.showinfo("Success", f"Image cleaned and saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    
    # Buttons
    clean_button = Button(root, text="Select and Clean Image", command=clean_image, 
                         width=20, height=2, font=("Arial", 12))
    clean_button.pack(pady=10)
    
    csv_button = Button(root, text="Sort and Export to CSV", command=sort_and_export_csv, 
                       width=20, height=2, font=("Arial", 12))
    csv_button.pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    process_image_gui()
    sys.exit() 
