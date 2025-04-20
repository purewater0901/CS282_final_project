import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Union, List

# Custom dataset for deepfake detection
class DeepfakeDataset(Dataset):
    """Custom dataset for testing deepfake detection models with customizable class folders"""
    def __init__(
        self,
        root_dir: str,
        real_folder: Union[str, List[str]] = 'Real',
        fake_folder: Union[str, List[str]] = 'Fake',
        transform=None,
        processor=None
    ):
        """
        Args:
            root_dir (str): Root directory containing class folders
            real_folder (str): Name of the folder containing real images
            fake_folder (str): Name of the folder containing fake images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.processor = processor
        self.class_folders = {
            0: real_folder,  # 0 = real
            1: fake_folder,  # 1 = fake
        }

        self.samples = []
        self.load_samples()

    def load_samples(self):
        """Load all image paths and their corresponding labels"""
        for class_idx, folder_list in self.class_folders.items():
             for folder_name in folder_list:
                 class_dir = os.path.join(self.root_dir, folder_name)
                 if not os.path.exists(class_dir):
                     raise FileNotFoundError(f"Directory not found: {class_dir}")
 
                 # Add all valid images from this class folder
                 for img_name in os.listdir(class_dir):
                     if img_name.lower().endswith(('.png')): # We restrict png format, in order to avoid overfitting to the difference in format
                         img_path = os.path.join(class_dir, img_name)
                         self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            elif self.processor:
                #image = self.processor(image=image, return_tensors="pt")["pixel_values"].squeeze(0)
                image = self.processor(image)
                # processor returns dictionary, so reduce dimension here

            return image, label, img_path

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the label
            placeholder = torch.zeros((3, 299, 299))
            return placeholder, label, img_path

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    real_folder = ['Real_1k_split/Train']
    fake_folder = ['firefly_split/Train']

    dataset = DeepfakeDataset(
        root_dir=data_dir,
        real_folder=real_folder,
        fake_folder=fake_folder,
    )

    image, label, img_path = dataset[0]
    print("Dataset Size: ", len(dataset))
    print("Image Label: ", label)
    print("Image path: ", img_path)