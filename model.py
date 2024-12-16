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
from torchsummary import summary  # Import summary function
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
    def __init__(self, image_dir, caption_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
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
        
        caption_tokens = self.tokenizer.encode(caption)  # Tokenize caption
        return image, torch.tensor(caption_tokens, dtype=torch.long)


class CustomModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)

        self.conv_output_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(512, output_size)

    def initialize_fc1(self, image_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            x = torch.relu(self.conv1(dummy_input))
            x = torch.relu(self.conv2(x))
            self.conv_output_size = x.numel()
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


# Training and Validation with Metrics
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
        all_train_preds, all_train_labels = [], []

        # Training loop
        for images, captions in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            counter += 1
            if counter > 100:
                counter = 0
                time.sleep(30)
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            outputs = model(images, captions)

            # Compute loss and accumulate
            loss = criterion(outputs, captions.view(-1))
            total_train_loss += loss.item()

            # Backward pass and optimizer step
            loss.backward()

            # Collect predictions and labels for metrics
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(captions.view(-1).cpu().numpy())

        optimizer.step()  # Apply accumulated gradients
        print("optimizing......")
        optimizer.zero_grad()  # Reset gradients after epoch

        # Perform learning rate scheduling
        scheduler.step()

        # Compute average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # Calculate metrics for training
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_labels = [], []

        # Validation loop
        with torch.no_grad():
            for images, captions in tqdm(val_loader, desc="Validating", leave=False):
                images, captions = images.to(device), captions.to(device)
                outputs = model(images, captions)
                loss = criterion(outputs, captions.view(-1))
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(captions.view(-1).cpu().numpy())

        # Calculate metrics for validation
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

        # Record metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1_score': train_f1,
            'val_f1_score': val_f1
        })

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, "
              f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, "
              f"Train F1-Score: {train_f1:.4f}, Val F1-Score: {val_f1:.4f}")
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
    image_dir = r"C:\Users\Nischal\Downloads\segmentation\captioned_images"
    caption_file = r"C:\Users\Nischal\Downloads\segmentation\output.txt"
    with open(caption_file, 'r', encoding='utf-8') as file:
        captions = [line.strip().split('|')[1].strip() for line in file.readlines()]

    tokenizer = CustomTokenizer(captions)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    # Print the model summary before training
    print("\nModel Summary:")
    summary(model, input_size=(3, 64, 64))

    # Setup optimizer and scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print("training model")

    # Train and validate the model
    metrics = train_and_validate(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device=device)

    # Save model, tokenizer, and metrics
    torch.save(model.state_dict(), "image_caption_model.pth")
    torch.save(tokenizer.vocab, "tokenizer_vocab.pth")
    save_metrics_to_csv(metrics, "training_metrics.csv")
