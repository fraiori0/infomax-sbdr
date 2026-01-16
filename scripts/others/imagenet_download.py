from datasets import load_dataset
import os

# Define a directory to store the dataset
output_dir = "~/research" # Choose a suitable path on your workstation
os.makedirs(output_dir, exist_ok=True)

print("Downloading ImageNet-1K train split...")
train_dataset = load_dataset('imagenet-1k', split='train', cache_dir=output_dir)
print(f"Train split downloaded to: {os.path.join(output_dir, 'imagenet-1k')}")

print("Downloading ImageNet-1K validation split...")
val_dataset = load_dataset('imagenet-1k', split='validation', cache_dir=output_dir)
print(f"Validation split downloaded to: {os.path.join(output_dir, 'imagenet-1k')}")

print("ImageNet-1K download complete.")