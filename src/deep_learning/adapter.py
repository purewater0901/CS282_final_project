# Loading Libraries
import os
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor

import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # Import tqdm for progress bars
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from IPython.display import display

from huggingface_hub import hf_hub_download

# from adapters import AutoAdapterModel
# from transformers import CLIPImageProcessor
# from transformers import CLIPVisionModel

from deepfake_detection.model.dfdet import DeepfakeDetectionModel
from deepfake_detection.config import Config
# from transformers.adapters import AdapterConfig

# Wandb
import wandb

import warnings
warnings.filterwarnings("ignore")

# Custom dataset for deepfake detection
class DeepfakeDataset(Dataset):
    """Custom dataset for testing deepfake detection models with customizable class folders"""
    def __init__(
        self, 
        root_dir: str, 
        real_folder: str = 'Real', 
        fake_folder: str = 'Fake',
        transform=None,
        processor=None
    ):
        """
        Args:
            root_dir (str): Root directory containing class folders
            real_folder (str): Name of the folder containing real images
            fake_folder (str): Name of the folder containing fake images
            transform (callable, optional): Optional transform to be applied on images, necessary if you create models via timm library
            processor (callable, optional): Pre-trained processor, necessary if you create models via Transformer library
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
        for class_idx, folder_name in self.class_folders.items():
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
                image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0) 
                # processor returns dictionary, so reduce dimension here
                
            return image, label, img_path
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the label
            placeholder = torch.zeros((3, 299, 299))
            return placeholder, label, img_path

class FineTuner():
    """
    A Class of fine-tuning.
            
    Args:
        data_dir (str): Directory containing 'train' and 'val' subdirectories, 
                        each with 'real' and 'fake' subdirectories
        real_folder:
        fake_folder:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        use_wandb (boolean): Use wandb's logging 
        model: Optional model if you load your model from transformer library
        processor: Optional processor if you load your model from transformer library - it comes with a pre-trained processor
    
    """
    def __init__(self, model_name, data_dir, real_folder, fake_folder, num_epochs, batch_size, learning_rate, use_wandb = False, model = None, processor = None):
        self.model_name = model_name
        
        # Load the model using timm if the model is None
        if model == None:
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=2)
            print("Loaded ", model_name, " for fine-tuning")
        else:
            self.model = model

        self.data_dir = data_dir
        self.real_folder = real_folder
        self.fake_folder = fake_folder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.test_fake_folder = None
        self.processor = processor

    def set_TestFolder(self, test_fake_folder):
        self.test_fake_folder = test_fake_folder

    def get_Train_Val_loader(self):

        if self.processor:
            train_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder= os.path.join(self.real_folder, 'Train'), 
                fake_folder= os.path.join(self.fake_folder, 'Train'),
                processor = self.processor
            )
        
            val_dataset = DeepfakeDataset(
                root_dir= self.data_dir,
                real_folder= os.path.join(self.real_folder, 'Validation'), 
                fake_folder= os.path.join(self.fake_folder, 'Validation'),
                processor = self.processor
            )
            
        else:
            # Get the config file from timm
            config = resolve_data_config({}, model=self.model)
            base_transform = create_transform(**config)
            # Data augmentation and normalization for training
            train_transform_list = base_transform.transforms
            train_transform_list.append(transforms.RandomHorizontalFlip())
            train_transform_list.append(transforms.RandomRotation(10))
            
            brightness = np.random.uniform(0.05, 0.2)  # Random value between 0.05 and 0.2... you can change if you want
            contrast = np.random.uniform(0.05, 0.2)
            saturation = np.random.uniform(0.05, 0.2)
            hue = np.random.uniform(0, 0.1)  # Hue is typically smaller values
            train_transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue = hue))
                    
            train_transform = transforms.Compose(train_transform_list)
            val_transform = base_transform
            
            # Create datasets
            train_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder= os.path.join(self.real_folder, 'Train'), 
                fake_folder= os.path.join(self.fake_folder, 'Train'),
                transform=train_transform
            )
            
            val_dataset = DeepfakeDataset(
                root_dir= self.data_dir,
                real_folder= os.path.join(self.real_folder, 'Validation'), 
                fake_folder= os.path.join(self.fake_folder, 'Validation'),
                transform=val_transform,
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return train_loader, val_loader

    def get_Test_loader(self, test_folder):
        target = test_folder
        if self.processor:
            # Create dataset
            dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder= os.path.join(self.real_folder, 'Test'), 
                fake_folder= os.path.join(target, 'Test'),
                processor = self.processor
            )
        else:
            # Get the config file from timm
            config = resolve_data_config({}, model=self.model)
            base_transform = create_transform(**config)
            
            
            # Create dataset
            dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder= os.path.join(self.real_folder, 'Test'), 
                fake_folder= os.path.join(target, 'Test'),
                transform=base_transform
            )
            
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return data_loader

    def Tune(self):
        # function to override
        pass
        

    def Evaluation(self, test_folder):
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get dataloader
        test_loader = self.get_Test_loader(test_folder)
        print('\n\n----- Test on ',test_folder,'-----')
        print(f"Device: {device}")
        
        
        # Set model to evaluation mode
        self.model.eval()
        model = self.model.to(device)
        
        # Lists to store results
        all_preds = []
        all_probs = []
        all_labels = []
        all_paths = []
        confidence_threshold = 0.5
        
        # Run inference
        with torch.no_grad():
            for inputs, labels, paths in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Convert outputs to probabilities
                probs = torch.softmax(outputs.float(), dim=1).cpu().numpy() # BF16 to float()
                
                # Convert to binary predictions using threshold
                preds = (probs[:,1] >= confidence_threshold).astype(int)
                
                # Store results
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Classification report
        report = classification_report(all_labels, all_preds, 
                                      target_names=['Real', 'Fake'], 
                                      output_dict=True)

        test_acc = (all_preds == all_labels).mean()

        print(f"\n\n{'-'*50}")
        print(f"Test Result Summary:")
        print(f"{'-'*50}")
        print(f"Test accuracy: {test_acc:.4f}")

        # Visualize the result ------------------------------------------------
        report_df = pd.DataFrame(report)
        display(report_df)

        cm = confusion_matrix(all_labels, all_preds)
        plt.subplot(2, 2, 4)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Real', 'Fake'], rotation=45)
        plt.yticks(tick_marks, ['Real', 'Fake'])
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.show()

        # Logging
        if self.use_wandb:
            
            all_probs = np.array(all_probs)
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:,1])
            wandb.log({
                "test_dataset": test_folder,
                "test_accuracy": test_acc,
                # Confusion Matrix
                "test_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=['Real', 'Fake']
                ),
                # ROC curve
                "test_roc_curve": wandb.plot.roc_curve(
                    y_true= all_labels,
                    y_probas= all_probs,
                    labels=['Real', 'Fake']
                )
            })
            
        return report_df

    def log_wandb_train(
            self,
            all_labels,
            all_preds,
            all_probs,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            optimizer,
        ):
        if self.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
                    
            # Extra logging for the last epoch
            if epoch == self.num_epochs - 1 and self.use_wandb:
                # Confusion Matrix
                cm = confusion_matrix(all_labels, all_preds)
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels,
                        preds=all_preds,
                        class_names=['Real', 'Fake']
                    )
                })
    
                # ROC curve - use probs instead of preds
                all_probs = np.array(all_probs)
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:,1])
                wandb.log({
                    "roc_curve": wandb.plot.roc_curve(
                        y_true= all_labels,
                        y_probas= all_probs,
                        labels=['Real', 'Fake']
                    )
                })
        
    def Experiment(self, wandb_run_name):
        # Initilize Wandb
        if self.use_wandb:
            wandb.init(project='Fine-Tuning Experiment', name=wandb_run_name)
        
            # Log the experimental setting
            wandb.config.update({
                "model": self.model_name,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "fine_tuning_type": "full",
                "dataset_dir": self.data_dir,
                "real_folder": self.real_folder,
                "fake_folder": self.fake_folder
            })
    
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.log({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
                "percent_trainable": 100 * trainable_params / total_params
            })
            
            # Log the model itself, (don't know if we need this or not)
            # wandb.watch(self.model, log="all", log_freq=100)
    
        # Fine-tune the model
        tuned_model = self.Tune()
        report_df_seen = self.Evaluation(self.fake_folder)
        if self.test_fake_folder != None:
            report_df_unseen = self.Evaluation(self.test_fake_folder)
    
        if self.use_wandb:
            model_artifact = wandb.Artifact(
                name=f"{self.method_name}-{self.model_name}", 
                type="model"
            )
            #model_artifact.add_file(model_save_path)
            wandb.log_artifact(model_artifact)
            wandb.finish()

        return tuned_model
    
    class FullFT(FineTuner):
    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate,
        use_wandb= False,
        model = None,
        processor = None):
        super().__init__(model_name, data_dir, real_folder, fake_folder, num_epochs, batch_size, learning_rate, use_wandb, model, processor)
        self.method_name = 'Full_FT' # Just name your method for the record

    def Tune(self):
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
        # Load the model
        #model = timm.create_model(self.model_name, pretrained=True, num_classes=2)
        self.model = self.model.to(device)
        model_save_path= f"{self.model_name}.pth"
                   
        # Get data loader for training and validation
        train_loader, val_loader = self.get_Train_Val_loader()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_acc = 0.0
        
        # Create a tqdm progress bar for epochs
        epoch_loop = tqdm(range(self.num_epochs), desc="Training Progress", unit="epoch")
        for epoch in epoch_loop:
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
    
            for inputs, labels, pathes in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs).float()
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            
            # Calculate epoch metrics
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            
            with torch.no_grad():
                for inputs, labels, pathes in val_loader: #val_loop:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    probs = torch.nn.functional.softmax(outputs.float(), dim=1)
                    _, predicted = torch.max(outputs.float(), 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            print(f"\n\n{'-'*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"{'-'*50}")
            # Save the best model
            if val_acc >= best_val_acc:
                torch.save(self.model.state_dict(), model_save_path)
                print(f"✅ New best model saved! Validation accuracy: {val_acc:.4f} (previous best: {best_val_acc:.4f})")
                best_val_acc = val_acc
            
            # Also save a checkpoint every 5 epoch
            if (epoch + 1) % 5 == 0: 
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            print(f"Training:   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} ({correct}/{total})")
            print(f"Validation: Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} ({val_correct}/{val_total})")
            
            # Calculate and display improvement or regression
            if epoch > 0:
                train_loss_change = train_loss - train_losses[-2]
                train_acc_change = train_acc - train_accuracies[-2]
                val_loss_change = val_loss - val_losses[-2]
                val_acc_change = val_acc - val_accuracies[-2]
                
                print(f"Changes from previous epoch:")
                print(f"  Train Loss: {train_loss_change:+.4f} | Train Acc: {train_acc_change:+.4f}")
                print(f"  Val Loss: {val_loss_change:+.4f} | Val Acc: {val_acc_change:+.4f}")
            
            # Display current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            #Load the best model
            print(f"Loading best model from {model_save_path}")
            self.model.load_state_dict(torch.load(model_save_path))
            
            # Log in wandb
            self.log_wandb_train(
                all_labels,
                all_preds,
                all_probs,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                optimizer
            )
        
        # Final summary at the end of training
        tqdm.write(f"\nTraining completed after {self.num_epochs} epochs")
        tqdm.write(f"Best validation accuracy: {best_val_acc:.4f}")
    
        return self.model
    
    
    class AdapterTuner(FineTuner):
    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate,
        use_wandb= False,
        model = None,
        processor = None):
        super().__init__(model_name, data_dir, real_folder, fake_folder, num_epochs, batch_size, learning_rate, use_wandb, model, processor)
        self.method_name = 'Adapter'
        
        # self.pre_adapter = PreAdapter()
        # self.pre_adapter.apply(self.init_weights)

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         nn.init.constant_(m.bias, 0)

    def Tune(self):
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
        # Load the model
        #model = timm.create_model(self.model_name, pretrained=True, num_classes=2)
        self.model = self.model.to(device)
        # self.pre_adapter = self.pre_adapter.to(device)
        model_save_path= f"{self.model_name}_Adapter.pth"
                   
        # Get data loader for training and validation
        train_loader, val_loader = self.get_Train_Val_loader()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        # # Define the optimizer to only update parameters of adapters
        # for param in model.parameters():
        #     param.requires_grad = False

        # for n, p in model.named_parameters():
        #     if "adapter" in n:
        #         p.requires_grad = True
            
        optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_acc = 0.0
        
        # Create a tqdm progress bar for epochs
        epoch_loop = tqdm(range(self.num_epochs), desc="Training Progress", unit="epoch")
        for epoch in epoch_loop:
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
    
            for inputs, labels, pathes in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # inputs = processor(images=inputs, return_tensors="pt").to(device)
                # pixel_values = processor(images=inputs, return_tensors="pt").to(device)["pixel_values"]
                # inputs = {"pixel_values": inputs}
                
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # added pre_adapter
                # inputs = self.pre_adapter(inputs)
                # outputs = self.model(inputs).float()
                # outputs = self.model(pixel_values=pixel_values).logits
                # outputs = self.model(**inputs).logits
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                # running_loss += loss.item() * inputs["pixel_values"].size(0)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            
            # Calculate epoch metrics
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            
            with torch.no_grad():
                for inputs, labels, pathes in val_loader: #val_loop:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # inputs = processor(images=inputs, return_tensors="pt").to(device)
                    # pixel_values = processor(images=inputs, return_tensors="pt").to(device)["pixel_values"]
                    # inputs = {"pixel_values": inputs}

                    # added pre_adapter
                    # inputs = self.pre_adapter(inputs)
                    # outputs = self.model(inputs)
                    # outputs = self.model(pixel_values=pixel_values).logits
                    # outputs = self.model(**inputs).logits
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # val_running_loss += loss.item() * inputs["pixel_values"].size(0)
                    val_running_loss += loss.item() * inputs.size(0)
                    probs = torch.nn.functional.softmax(outputs.float(), dim=1)
                    _, predicted = torch.max(outputs.float(), 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            print(f"\n\n{'-'*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"{'-'*50}")
            # Save the best model
            if val_acc >= best_val_acc:
                torch.save(self.model.state_dict(), model_save_path)
                print(f"✅ New best model saved! Validation accuracy: {val_acc:.4f} (previous best: {best_val_acc:.4f})")
                best_val_acc = val_acc
            
            # Also save a checkpoint every 5 epoch
            if (epoch + 1) % 5 == 0: 
                checkpoint_path = f"checkpoint_epoch_{epoch+1}_Adapter.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            print(f"Training:   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} ({correct}/{total})")
            print(f"Validation: Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} ({val_correct}/{val_total})")
            
            # Calculate and display improvement or regression
            if epoch > 0:
                train_loss_change = train_loss - train_losses[-2]
                train_acc_change = train_acc - train_accuracies[-2]
                val_loss_change = val_loss - val_losses[-2]
                val_acc_change = val_acc - val_accuracies[-2]
                
                print(f"Changes from previous epoch:")
                print(f"  Train Loss: {train_loss_change:+.4f} | Train Acc: {train_acc_change:+.4f}")
                print(f"  Val Loss: {val_loss_change:+.4f} | Val Acc: {val_acc_change:+.4f}")
            
            # Display current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            #Load the best model
            print(f"Loading best model from {model_save_path}")
            self.model.load_state_dict(torch.load(model_save_path))
            
            # Log in wandb
            self.log_wandb_train(
                all_labels,
                all_preds,
                all_probs,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                optimizer
            )
        
        # Final summary at the end of training
        tqdm.write(f"\nTraining completed after {self.num_epochs} epochs")
        tqdm.write(f"Best validation accuracy: {best_val_acc:.4f}")
    
        return self.model
    
class AdapterTuner(FineTuner):
    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate,
        use_wandb= False,
        model = None,
        processor = None):
        super().__init__(model_name, data_dir, real_folder, fake_folder, num_epochs, batch_size, learning_rate, use_wandb, model, processor)
        self.method_name = 'Adapter'
        
        # self.pre_adapter = PreAdapter()
        # self.pre_adapter.apply(self.init_weights)

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         nn.init.constant_(m.bias, 0)

    def Tune(self):
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
        # Load the model
        #model = timm.create_model(self.model_name, pretrained=True, num_classes=2)
        self.model = self.model.to(device)
        # self.pre_adapter = self.pre_adapter.to(device)
        model_save_path= f"{self.model_name}_Adapter.pth"
                   
        # Get data loader for training and validation
        train_loader, val_loader = self.get_Train_Val_loader()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        # # Define the optimizer to only update parameters of adapters
        # for param in model.parameters():
        #     param.requires_grad = False

        # for n, p in model.named_parameters():
        #     if "adapter" in n:
        #         p.requires_grad = True
            
        optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_acc = 0.0
        
        # Create a tqdm progress bar for epochs
        epoch_loop = tqdm(range(self.num_epochs), desc="Training Progress", unit="epoch")
        for epoch in epoch_loop:
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
    
            for inputs, labels, pathes in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # inputs = processor(images=inputs, return_tensors="pt").to(device)
                # pixel_values = processor(images=inputs, return_tensors="pt").to(device)["pixel_values"]
                # inputs = {"pixel_values": inputs}
                
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # added pre_adapter
                # inputs = self.pre_adapter(inputs)
                # outputs = self.model(inputs).float()
                # outputs = self.model(pixel_values=pixel_values).logits
                # outputs = self.model(**inputs).logits
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                # running_loss += loss.item() * inputs["pixel_values"].size(0)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            
            # Calculate epoch metrics
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            
            with torch.no_grad():
                for inputs, labels, pathes in val_loader: #val_loop:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # inputs = processor(images=inputs, return_tensors="pt").to(device)
                    # pixel_values = processor(images=inputs, return_tensors="pt").to(device)["pixel_values"]
                    # inputs = {"pixel_values": inputs}

                    # added pre_adapter
                    # inputs = self.pre_adapter(inputs)
                    # outputs = self.model(inputs)
                    # outputs = self.model(pixel_values=pixel_values).logits
                    # outputs = self.model(**inputs).logits
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # val_running_loss += loss.item() * inputs["pixel_values"].size(0)
                    val_running_loss += loss.item() * inputs.size(0)
                    probs = torch.nn.functional.softmax(outputs.float(), dim=1)
                    _, predicted = torch.max(outputs.float(), 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            print(f"\n\n{'-'*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"{'-'*50}")
            # Save the best model
            if val_acc >= best_val_acc:
                torch.save(self.model.state_dict(), model_save_path)
                print(f"✅ New best model saved! Validation accuracy: {val_acc:.4f} (previous best: {best_val_acc:.4f})")
                best_val_acc = val_acc
            
            # Also save a checkpoint every 5 epoch
            if (epoch + 1) % 5 == 0: 
                checkpoint_path = f"checkpoint_epoch_{epoch+1}_Adapter.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            print(f"Training:   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} ({correct}/{total})")
            print(f"Validation: Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} ({val_correct}/{val_total})")
            
            # Calculate and display improvement or regression
            if epoch > 0:
                train_loss_change = train_loss - train_losses[-2]
                train_acc_change = train_acc - train_accuracies[-2]
                val_loss_change = val_loss - val_losses[-2]
                val_acc_change = val_acc - val_accuracies[-2]
                
                print(f"Changes from previous epoch:")
                print(f"  Train Loss: {train_loss_change:+.4f} | Train Acc: {train_acc_change:+.4f}")
                print(f"  Val Loss: {val_loss_change:+.4f} | Val Acc: {val_acc_change:+.4f}")
            
            # Display current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            #Load the best model
            print(f"Loading best model from {model_save_path}")
            self.model.load_state_dict(torch.load(model_save_path))
            
            # Log in wandb
            self.log_wandb_train(
                all_labels,
                all_preds,
                all_probs,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                optimizer
            )
        
        # Final summary at the end of training
        tqdm.write(f"\nTraining completed after {self.num_epochs} epochs")
        tqdm.write(f"Best validation accuracy: {best_val_acc:.4f}")
    
        return self.model
    
# Testing to load pre-deepfake trained model
model_path = "./weights/model.ckpt"
if not os.path.exists(model_path):
    print("Downloading model")
    os.makedirs("weights", exist_ok=True)
    os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.ckpt -O {model_path}")
    
ckpt = torch.load(model_path, map_location="cpu")

base_model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
base_model.load_state_dict(ckpt["state_dict"])


# Save the classifier layer
classifier = base_model.model.linear

# Remove it from the model so it's not run inside forward()
base_model.model.linear = nn.Identity()



class AdapterWrapper(nn.Module):
    def __init__(self, base_model, classifier, hidden_dim=1024, bottleneck=1024*2):
        super().__init__()
        self.base_model = base_model
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, hidden_dim)
        )
        self.classifier = classifier

    def forward(self, x):
        features = self.base_model(x)         # HeadOutput object
        x_tensor = features.features          
        adapted = x_tensor + self.adapter(x_tensor)  # residual connection
        logits = self.classifier(adapted)
        return logits


model = AdapterWrapper(base_model, classifier)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir='./huggingface')


for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = False

for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)

config = {
    'model_name': 'DeepfakeDetectionModel_Adapters_2048',
    "data_dir":"./",
    "real_folder" : 'Real_split',
    "fake_folder" : 'All_fakes_split',
    "num_epochs":20,
    "batch_size":16,
    "learning_rate":0.0005,
    'use_wandb':False
}

adapter_tuner = AdapterTuner(model=model, processor=processor, **config)
tuned_model = adapter_tuner.Experiment('adapter_experiment')