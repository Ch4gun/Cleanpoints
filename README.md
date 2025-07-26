# Cleanpoints

A Python GUI tool for cleaning leaderboard images and extracting data using OCR (Tesseract). Perfect for processing gaming leaderboards, clan rankings, and other tabular image data.

## Features

- **Image Cleaning**: Removes crown icons and associated numbers from leaderboard images
- **OCR Processing**: Extracts text and numbers from cleaned images
- **CSV Export**: Converts extracted data to CSV format with customizable delimiters
- **Text Sorting**: Sorts extracted data by points in descending order
- **User-Friendly GUI**: Simple Windows interface for easy operation

## Requirements

- Python 3.7+
- Tesseract-OCR
- Required Python packages:
  ```
  pip install opencv-python pillow numpy pytesseract
  ```

## Installation

1. Install Tesseract-OCR from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install required Python packages:
   ```
   pip install opencv-python pillow numpy pytesseract
   ```
3. Update the Tesseract path in `main.py` if needed:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```
2. Choose your operation:
   - **Select and Clean Image**: Clean crown icons and numbers from leaderboard images
   - **Sort and Export to CSV**: Process text files and export sorted data to CSV

## Function Documentation

### Core Functions

#### `remove_crowns_and_numbers(image_path, left_crop=400)`
**Purpose**: Removes crown icons and their associated numbers from leaderboard images.

**How it works**:
- Detects orange/yellow colored crown elements using HSV color filtering
- Uses OCR to identify numbers below crowns
- Removes crowns and their associated numbers with white rectangles
- Preserves nicknames and points data

**Parameters**:
- `image_path`: Path to the input image
- `left_crop`: Number of pixels to process from the left (default: 400)

**Returns**: Processed image with crowns and numbers removed

#### `extract_text_from_image(image_path)`
**Purpose**: Extracts clean text from images using OCR.

**How it works**:
- Runs Tesseract OCR with optimized settings
- Cleans up OCR artifacts and unwanted characters
- Filters out noise and short text fragments
- Returns cleaned text lines

**Parameters**:
- `image_path`: Path to the input image

**Returns**: Cleaned text string

#### `correct_ocr_result(input_file_path, output_file_path)`
**Purpose**: Processes OCR results and converts to CSV format.

**How it works**:
- Reads text file with OCR results
- Pairs nicknames with their corresponding numbers
- Applies spelling corrections for common OCR errors
- Exports to CSV format

**Parameters**:
- `input_file_path`: Path to OCR result text file
- `output_file_path`: Path for output CSV file

**Returns**: Number of processed entries

#### `extract_pairs_by_position(image_path)`
**Purpose**: Extracts nickname-number pairs from images using positional analysis.

**How it works**:
- Groups OCR text by vertical position (lines)
- Identifies nicknames vs numbers based on content
- Matches numbers to nicknames based on proximity
- Filters out OCR artifacts

**Parameters**:
- `image_path`: Path to the input image

**Returns**: List of [nickname, points] pairs

### GUI Functions

#### `process_image_gui()`
**Purpose**: Main GUI application with all user interface elements.

**Features**:
- Image format selection (PNG/JPEG)
- CSV delimiter selection (Tab/Comma)
- File selection dialogs
- Progress feedback and error handling

#### `sort_and_export_csv()`
**Purpose**: GUI function for sorting and exporting text data to CSV.

**How it works**:
- Reads text file with nickname-number pairs
- Sorts by points in descending order
- Exports to CSV with selected delimiter
- Provides success/error feedback

#### `clean_image()`
**Purpose**: GUI function for cleaning leaderboard images.

**How it works**:
- Opens file dialog for image selection
- Processes image using `remove_crowns_and_numbers()`
- Saves cleaned image in selected format
- Provides success/error feedback

### Utility Functions

#### `load_mapping()` / `save_mapping(mapping)`
**Purpose**: Manages nickname mapping data for OCR correction.

**How it works**:
- Loads/saves mapping from/to JSON file
- Maps OCR-detected nicknames to correct names
- Used for improving OCR accuracy

#### `manage_mapping_ui(mapping, root, ...)`
**Purpose**: GUI for managing nickname mappings.

**Features**:
- Add/remove nickname mappings
- Export mappings to CSV
- Visual table interface

#### `ocr_lines(image, psm=6)`
**Purpose**: Runs Tesseract OCR on image and returns text lines.

**Parameters**:
- `image`: PIL Image object
- `psm`: Page segmentation mode (default: 6 for uniform text block)

**Returns**: List of text lines

## Configuration

### Key Constants

```python
NICKNAME_CROP_RIGHT = 350  # Right edge of nickname column
POINTS_CROP_LEFT = -200    # Left edge of points column (from right)
LEFT_CROP = 400           # Pixels to process from left
```

### Color Detection

The crown detection uses HSV color filtering:
- **Lower bound**: `[15, 120, 120]` (orange/yellow)
- **Upper bound**: `[25, 255, 255]` (orange/yellow)

## File Structure

```
Cleanpoints/
├── main.py              # Main application file
├── README.md           # This documentation
├── requirements.txt    # Python dependencies
└── .gitignore         # Git ignore file
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Update the path in `main.py`
2. **Poor OCR results**: Ensure images are high quality and well-lit
3. **Crowns not detected**: Adjust HSV color ranges in `remove_crowns_and_numbers()`
4. **Wrong text extraction**: Check image orientation and text clarity

### Performance Tips

- Use high-resolution images for better OCR results
- Ensure good contrast between text and background
- Process images in batches for efficiency

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository. 
