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

from deep_learning.fine_tuner.py import FineTuner

# Wandb
import wandb

import warnings
warnings.filterwarnings("ignore")


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