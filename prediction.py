import torch
from torchvision import transforms
from PIL import Image
from model2 import CustomModel, CustomTokenizer  # Import your model and tokenizer
import pyttsx3  # TTS library

def predict_caption(image_path, model, tokenizer, device):
    # Ensure the model is in evaluation mode
    model.eval()

    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        # Create a dummy caption tensor for the forward pass
        dummy_caption = torch.zeros((1, 10), dtype=torch.long).to(device)  # Adjust length as needed
        outputs = model(image, dummy_caption)
    
    # Get predicted token indices
    predicted_tokens = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Decode tokens into text
    predicted_caption = tokenizer.decode(predicted_tokens)
    return predicted_caption

def speak_caption(caption):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()

if __name__ == "__main__":
    # Paths and device setup
    image_path = r"C:\Users\Nischal\Downloads\segmentation\captioned_images\word_6_20241105_112020_766024.png"  # Update this with your sample image path
    model_path = "image_caption_model.pth"
    vocab_path = "tokenizer_vocab.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    vocab = torch.load(vocab_path)
    tokenizer = CustomTokenizer([])
    tokenizer.vocab = vocab
    tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}

    # Load and setup the model
    vocab_size = len(tokenizer.vocab)
    output_size = vocab_size
    model = CustomModel(vocab_size, output_size)
    image_shape = (3, 64, 64)
    model.initialize_fc1(image_shape)  # Ensure FC1 is initialized
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Predict caption for the sample image
    try:
        predicted_caption = predict_caption(image_path, model, tokenizer, device)
        print(f"Predicted Caption: {predicted_caption}")
        
        # Speak the predicted caption
        speak_caption(predicted_caption)
    except Exception as e:
        print(f"Error during prediction: {e}")
