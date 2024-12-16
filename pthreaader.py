import torch

# Path to the .pth file
model_path = r"C:\Users\Nischal\Downloads\segmentation\tokenizer_vocab.pth"  # Replace with your .pth file path

# Load the state dictionary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Print the keys of the state dictionary
print("Keys in the state dictionary:")
for key in state_dict.keys():
    print(key)

# If you want to inspect a specific layer's weights, you can do so:
print("\nWeights of a specific layer (e.g., 'conv1.weight'):")
if 'conv1.weight' in state_dict:
    print(state_dict['conv1.weight'])
