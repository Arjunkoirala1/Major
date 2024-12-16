import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model2 import CustomModel, CustomTokenizer
import pyttsx3

def preprocess_image(image, scaling_factor=2.9):
    """Scale, convert to grayscale, and threshold the image for easier contour detection."""
    width = int(image.shape[1] * scaling_factor)
    height = int(image.shape[0] * scaling_factor)
    enlarged_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    white_pixels = np.sum(blurred > 127)
    black_pixels = np.sum(blurred <= 127)
    
    if black_pixels > white_pixels:
        blurred = cv2.bitwise_not(blurred)
    
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return enlarged_image, thresh

def enhance_character_separation(thresh):
    """Use morphological operations to improve character separation in the thresholded image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated

def detect_contours(dilated):
    """Detect contours from the dilated image."""
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def split_connected_words(x, y, w, h, thresh):
    """Detect and split connected words based on vertical projection gaps."""
    roi = thresh[y:y+h, x:x+w]
    v_proj = np.sum(roi, axis=0)
    
    split_positions = []
    current_gap_start = None
    for i in range(len(v_proj)):
        if v_proj[i] == 0:
            if current_gap_start is None:
                current_gap_start = i
        else:
            if current_gap_start is not None:
                if i - current_gap_start > 10:
                    split_positions.append((current_gap_start, i))
                current_gap_start = None

    if not split_positions:
        return [(x, y, w, h)]
    else:
        split_boxes = []
        prev_x = 0
        for start, end in split_positions:
            split_boxes.append((x + prev_x, y, end - prev_x, h))
            prev_x = end
        if prev_x < w:
            split_boxes.append((x + prev_x, y, w - prev_x, h))
        return split_boxes

def filter_and_split_boxes(contours, thresh, min_width=20, max_width=400, min_height=20, max_height=200):
    """Filter contours by size and split boxes as needed."""
    word_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if min_width <= w <= max_width and min_height <= h <= max_height:
            split_boxes = split_connected_words(x, y, w, h, thresh)
            word_boxes.extend(split_boxes)
    return word_boxes

def adjust_large_boxes(word_boxes):
    """Adjust bounding boxes if their heights vary significantly."""
    adjusted_boxes = []
    for i in range(len(word_boxes) - 1):
        x1, y1, w1, h1 = word_boxes[i]
        x2, y2, w2, h2 = word_boxes[i + 1]

        if h1 > 2 * h2:
            split_height = h1 // 2
            adjusted_boxes.append((x1, y1, w1, split_height))
            adjusted_boxes.append((x1, y1 + split_height, w1, h1 - split_height))
        else:
            adjusted_boxes.append((x1, y1, w1, h1))
    
    adjusted_boxes.append(word_boxes[-1])
    return adjusted_boxes

def sort_boxes_by_position(word_boxes, line_threshold=20):
    """Sort bounding boxes by vertical and horizontal alignment."""
    word_boxes.sort(key=lambda box: (box[1], box[0]))

    sorted_lines = []
    current_line = [word_boxes[0]]
    
    for i in range(1, len(word_boxes)):
        x, y, w, h = word_boxes[i]
        _, prev_y, _, prev_h = word_boxes[i - 1]

        if abs(y - prev_y) <= line_threshold:
            current_line.append(word_boxes[i])
        else:
            sorted_lines.append(sorted(current_line, key=lambda box: box[0]))
            current_line = [word_boxes[i]]
    
    if current_line:
        sorted_lines.append(sorted(current_line, key=lambda box: box[0]))

    sorted_boxes = [box for line in sorted_lines for box in line]
    return sorted_boxes

def extract_text_from_image(image, word_boxes):
    """Extract text from word boxes by predicting each word."""
    # Set up tokenizer and model
    vocab_path = "tokenizer_vocab.pth"
    model_path = "image_caption_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = torch.load(vocab_path)
    tokenizer = CustomTokenizer([])  # Replace with your actual tokenizer
    tokenizer.vocab = vocab
    tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}

    vocab_size = len(tokenizer.vocab)
    model = CustomModel(vocab_size, vocab_size)  # Adjust based on your model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    captions = []
    for (x, y, w, h) in word_boxes:
        word_image = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(word_image)

        word_image = transform(pil_image).unsqueeze(0).to(device)
        
        # Dummy input for caption prediction (you should adjust according to your model)
        dummy_caption = torch.zeros((1, 10), dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(word_image, dummy_caption)
        
        predicted_tokens = torch.argmax(outputs, dim=1).cpu().numpy()
        predicted_caption = tokenizer.decode(predicted_tokens)
        captions.append(predicted_caption)

    return ' '.join(captions)

def speak_caption(caption):
    """Speak the predicted caption."""
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()

def main(image_path):
    """Main function to process the image, predict caption, and speak it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path.")
        return

    enlarged_image, thresh = preprocess_image(image)
    dilated = enhance_character_separation(thresh)
    contours = detect_contours(dilated)

    word_boxes = filter_and_split_boxes(contours, thresh)
    if not word_boxes:
        print("No word boxes detected.")
        return

    adjusted_boxes = adjust_large_boxes(word_boxes)
    sorted_word_boxes = sort_boxes_by_position(adjusted_boxes)

    predicted_caption = extract_text_from_image(enlarged_image, sorted_word_boxes)
    print(f"Predicted Caption: {predicted_caption}")
    speak_caption(predicted_caption)

# Run the code
IMAGE_PATH = r'C:\Users\Nischal\Downloads\segmentation\1.png'  # Replace with your image path
main(IMAGE_PATH)
