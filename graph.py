import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
df = pd.read_csv("training_metrics.csv")

# 1. Plotting Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 2. Plotting Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', marker='o')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 3. Plotting Training and Validation Precision
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_precision'], label='Train Precision', marker='o')
plt.plot(df['epoch'], df['val_precision'], label='Validation Precision', marker='x')
plt.title('Training and Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# 4. Plotting Training and Validation Recall
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_recall'], label='Train Recall', marker='o')
plt.plot(df['epoch'], df['val_recall'], label='Validation Recall', marker='x')
plt.title('Training and Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()

# 5. Plotting Training and Validation F1-Score
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_f1_score'], label='Train F1-Score', marker='o')
plt.plot(df['epoch'], df['val_f1_score'], label='Validation F1-Score', marker='x')
plt.title('Training and Validation F1-Score')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)
plt.show()
