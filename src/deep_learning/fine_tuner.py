import os
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from utils.data_loader import DeepfakeDataset


class FineTuner:
    """
    A Class of fine-tuning.
    """

    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate,
        model=None,
        processor=None,
        device = "cpu"
    ):
        self.model_name = model_name

        # Load the model using timm if the model is None
        if model == None:
            self.model = timm.create_model(
                self.model_name, pretrained=True, num_classes=2
            )
            print("Loaded ", model_name, " for fine-tuning")
        else:
            self.model = model

        # unfreeze all weights and biases
        self.unfreeze_all_layers()

        self.data_dir = data_dir
        self.real_folder = real_folder
        self.fake_folder = fake_folder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.test_fake_folder = None
        self.processor = processor
        self.device = device

        # Move the model to the specified device
        self.model = self.model.to(self.device)

    def set_TestFolder(self, test_fake_folder):
        self.test_fake_folder = test_fake_folder

    def get_Train_Val_loader(self):

        if self.processor:
            print('Using the pre-loaded processor...')
            train_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=os.path.join(self.real_folder, "Train"),
                fake_folder=os.path.join(self.fake_folder, "Train"),
                processor=self.processor,
            )

            val_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=os.path.join(self.real_folder, "Validation"),
                fake_folder=os.path.join(self.fake_folder, "Validation"),
                processor=self.processor,
            )
        else:
            print('No processor found, using timm transforms...')
            # Get the config file from timm
            config = resolve_data_config({}, model=self.model)
            base_transform = create_transform(**config)
            # Data augmentation and normalization for training
            train_transform_list = base_transform.transforms
            train_transform_list.append(transforms.RandomHorizontalFlip())
            train_transform_list.append(transforms.RandomRotation(10))

            brightness = np.random.uniform(
                0.05, 0.2
            )  # Random value between 0.05 and 0.2... you can change if you want
            contrast = np.random.uniform(0.05, 0.2)
            saturation = np.random.uniform(0.05, 0.2)
            hue = np.random.uniform(0, 0.1)  # Hue is typically smaller values
            train_transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )
            train_transform = transforms.Compose(train_transform_list)
            val_transform = base_transform

            # Create datasets
            train_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=os.path.join(self.real_folder, "Train"),
                fake_folder=os.path.join(self.fake_folder, "Train"),
                transform=train_transform,
            )
            val_dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=os.path.join(self.real_folder, "Validation"),
                fake_folder=os.path.join(self.fake_folder, "Validation"),
                transform=val_transform,
            )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return train_loader, val_loader

    def get_Test_loader(self, test_folder):
        target = test_folder
        if self.processor:
            # Create dataset
            dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=os.path.join(self.real_folder, "Test"),
                fake_folder=os.path.join(target, "Test"),
                processor=self.processor,
            )
        else:
            # Get the config file from timm
            config = resolve_data_config({}, model=self.model)
            base_transform = create_transform(**config)

            # Create dataset
            dataset = DeepfakeDataset(
                root_dir=self.data_dir,
                real_folder=os.path.join(self.real_folder, "Test"),
                fake_folder=os.path.join(target, "Test"),
                transform=base_transform,
            )

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return data_loader

    def freeze_all_layers(self):
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True
    
    def count_trainable_params(self):
        # Count trainable parameters
        c = 0
        for param in self.model.parameters():  
            if param.requires_grad:
                c += param.numel()
        return c

    def Tune(self):
        # function to override
        pass

    def Evaluation(self, test_folder):

        # Get dataloader
        test_loader = self.get_Test_loader(test_folder)
        print("\n\n----- Test on ", test_folder, "-----")
        print(f"Device: {self.device}")

        # Set model to evaluation mode
        self.model.eval()

        # Lists to store results
        all_preds = []
        all_probs = []
        all_labels = []
        all_paths = []
        confidence_threshold = 0.5

        # Run inference
        with torch.no_grad():
            for inputs, labels, paths in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(self.device)

                # Forward pass
                if self.model_name == "deepfake_detection":
                        outputs = self.model(inputs).logits_labels
                else:
                    outputs = self.model(inputs)

                # Convert outputs to probabilities
                probs = (
                    torch.softmax(outputs.float(), dim=1).cpu().numpy()
                )  # BF16 to float()

                # Convert to binary predictions using threshold
                preds = (probs[:, 1] >= confidence_threshold).astype(int)

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
        report = classification_report(
            all_labels, all_preds, target_names=["Real", "Fake"], output_dict=True
        )

        test_acc = (all_preds == all_labels).mean()

        print(f"\n\n{'-'*50}")
        print(f"Test Result Summary:")
        print(f"{'-'*50}")
        print(f"Test accuracy: {test_acc:.4f}")

        # Visualize the result ------------------------------------------------
        report_df = pd.DataFrame(report)
        #display(report_df)

        cm = confusion_matrix(all_labels, all_preds)
        plt.subplot(2, 2, 4)
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Real", "Fake"], rotation=45)
        plt.yticks(tick_marks, ["Real", "Fake"])

        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.show()

        # Logging
        all_probs = np.array(all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        wandb.log(
            {
                "test_dataset": test_folder,
                "test_accuracy": test_acc,
                # Confusion Matrix
                "test_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=["Real", "Fake"],
                    title="Test Confusion Matrix"
                ),
                # ROC curve
                "test_roc_curve": wandb.plot.roc_curve(
                    y_true=all_labels,
                    y_probas=all_probs, 
                    labels=["Real", "Fake"],
                    title = "Test ROC Curve"
                ),
            }
        )

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
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Extra logging for the last epoch
        if epoch == self.num_epochs - 1:
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels,
                        preds=all_preds,
                        class_names=["Real", "Fake"],
                        title="Training Confusion Matrix",
                    )
                }
            )

            # ROC curve - use probs instead of preds
            all_probs = np.array(all_probs)
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            wandb.log(
                {
                    "roc_curve": wandb.plot.roc_curve(
                        y_true=all_labels,
                        y_probas=all_probs,
                        labels=["Real", "Fake"],
                        title="Training ROC Curve",
                    )
                }
            )

    def Experiment(self):
        # Fine-tune the model
        tuned_model = self.Tune()
        report_df_seen = self.Evaluation(self.fake_folder)
        if self.test_fake_folder != None:
            report_df_unseen = self.Evaluation(self.test_fake_folder)
        
        # Log Wandb
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.count_trainable_params()
        wandb.log(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
                "percent_trainable": 100 * trainable_params / total_params,
            }
        )
        model_artifact = wandb.Artifact(name=f"{self.method_name}-{self.model_name}", type="model")
        wandb.log_artifact(model_artifact)
        wandb.finish()

        return tuned_model
