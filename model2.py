from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image
import os
import csv
import time 
scaler = torch.cuda.amp.GradScaler()

# Custom Tokenizer
class CustomTokenizer:
    def __init__(self, captions):
        self.vocab = self.create_vocab(captions)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def create_vocab(self, captions):
        vocab = {'<PAD>': 0, '<UNK>': 1}  # Adding special tokens
        for caption in captions:
            for word in caption.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def encode(self, text):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]
    
    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '<UNK>') for token in tokens])

# Custom Dataset for Image-Caption Pairs
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, transform=None, max_length=100):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        # Load image paths and captions
        self.image_paths = []
        self.captions = []
        with open(caption_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split('|')  # Split on '|'
            if len(parts) == 2:
                image_name, caption = parts[0].strip(), parts[1].strip()
                if not image_name.lower().endswith('.png'):
                    image_name += '.png'
                # Check for empty captions
                if caption.strip():  # Ensure caption is not empty
                    self.image_paths.append(image_name)
                    self.captions.append(caption)

        # Ensure valid image files
        self.image_paths = [img for img in self.image_paths if os.path.exists(os.path.join(self.image_dir, img))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        caption = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        # Tokenize caption and check for issues
        caption_tokens = self.tokenizer.encode(caption)
        if len(caption_tokens) == 0:
            raise ValueError(f"Empty tokenized caption for image: {image_path} and caption: '{caption}'")

        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)

        # Pad the caption to the max_length
        if len(caption_tokens) < self.max_length:
            padding = torch.zeros(self.max_length - len(caption_tokens), dtype=torch.long)
            caption_tokens = torch.cat([caption_tokens, padding])
        else:
            caption_tokens = caption_tokens[:self.max_length]

        return image, caption_tokens

class CustomModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)

        # Calculate dynamically based on a dummy input
        self.conv_output_size = None

        # Define fully connected layers (initialize later)
        self.fc1 = None
        self.fc2 = nn.Linear(512, output_size)

    def initialize_fc1(self, image_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)  # Batch size of 1, RGB image
            x = torch.relu(self.conv1(dummy_input))
            x = torch.relu(self.conv2(x))
            self.conv_output_size = x.numel()  # Flattened size
            self.fc1 = nn.Linear(self.conv_output_size + 128, 512)

    def forward(self, image, caption):
        if self.fc1 is None:
            raise ValueError("Model not initialized. Call `initialize_fc1` first with the input image shape.")
        
        # Process image
        x_img = torch.relu(self.conv1(image))
        x_img = torch.relu(self.conv2(x_img))
        x_img = x_img.view(x_img.size(0), -1)

        # Process caption
        x_cap = self.embedding(caption)
        x_cap, _ = self.lstm(x_cap)
        x_cap = x_cap[:, -1, :]  # Use the last LSTM output

        # Concatenate and pass through FC layers
        x = torch.cat([x_img, x_cap], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training and Validation
def train_and_validate(
    model, train_loader, val_loader, optimizer, scheduler, epochs, device
):
    counter = 0
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    metrics = []

    for epoch in range(epochs):
        print("training .......")
        model.train()
        total_train_loss = 0

        # Training loop
    for images, captions in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            images, captions = images.to(device), captions.to(device)
    
    # Forward pass
    with torch.cuda.amp.autocast():  # For mixed precision training
        outputs = model(images)
        
        # Debugging shapes
        print(f"Outputs shape: {outputs.shape}")  # Should be [batch_size, seq_len, num_classes]
        print(f"Captions shape: {captions.shape}")  # Should be [batch_size, seq_len]

        # Compute the loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))


            # Backward pass and optimizer step
        loss.backward()
        optimizer.step()  # Apply accumulated gradients
        print("optimizing......")
        optimizer.zero_grad()  # Reset gradients after epoch

        # Perform learning rate scheduling
        scheduler.step()

        # Compute average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        # Validation loop
        with torch.no_grad():
            for images, captions in tqdm(val_loader, desc="Validating", leave=False):
                images, captions = images.to(device), captions.to(device)
                outputs = model(images, captions)
                loss = criterion(outputs, captions.view(-1))
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(captions.view(-1).cpu().numpy())

        # Calculate metrics
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Record metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })


        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
             f"Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return metrics



# Save Metrics to CSV
def save_metrics_to_csv(metrics, filename):
    keys = metrics[0].keys()
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)

# Main Execution
if __name__ == "__main__":
    # Load captions from the caption file
    image_dir = r"C:\Users\Nischal\Downloads\segmentation\captioned_images"
    caption_file = r"C:\Users\Nischal\Downloads\segmentation\output.txt"
    with open(caption_file, 'r', encoding='utf-8') as file:
        captions = [line.strip().split('|')[1].strip() for line in file.readlines()]

    tokenizer = CustomTokenizer(captions)

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare dataset and dataloaders
    dataset = ImageCaptionDataset(image_dir, caption_file, tokenizer, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(f"len of dataset {len(dataset)}")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(len(train_loader))
    # Model setup
    vocab_size = len(tokenizer.vocab)
    output_size = vocab_size  # Output size is the size of vocabulary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    model = CustomModel(vocab_size, output_size)

    # Initialize the model with the input image shape
    image_shape = (3, 64, 64)  # RGB images of size 64x64
    model.initialize_fc1(image_shape)

    # Setup optimizer and scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print("training model")
    # Train and validate the model
    metrics = train_and_validate(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device=device)

    # Save model, tokenizer, and metrics
    torch.save(model.state_dict(), "image_caption_model.pth")
    torch.save(tokenizer.vocab, "tokenizer_vocab.pth")
    save_metrics_to_csv(metrics, "training_metrics.csv")
