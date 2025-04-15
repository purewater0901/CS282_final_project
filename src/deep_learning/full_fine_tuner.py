import os
from tqdm.auto import tqdm
import torch
from torch import nn, optim

from deep_learning.fine_tuner import FineTuner


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
        model=None,
        processor=None,
        device = "cpu",
        **kwargs
    ):
        super().__init__(
            model_name,
            data_dir,
            real_folder,
            fake_folder,
            num_epochs,
            batch_size,
            learning_rate,
            model,
            processor,
            device
        )
        self.method_name = "Full_FT"

    def Tune(self, optimizer=None, model_save_path=None):
        """
        Args:
            data_dir (str): Directory containing 'train' and 'val' subdirectories,
                            each with 'real' and 'fake' subdirectories
            real_folder:
            fake_folder:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
        """
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Set device
        print(f"Using device: {self.device}")

        if model_save_path is None:
            model_save_path = f"full_tune_{self.model_name}.pth"

        # Optimizer
        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Get data loader for training and validation
        train_loader, val_loader = self.get_Train_Val_loader()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_acc = 0.0

        # Create a tqdm progress bar for epochs
        epoch_loop = tqdm(
            range(self.num_epochs), desc="Training Progress", unit="epoch"
        )
        for epoch in epoch_loop:
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels, pathes in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                if self.model_name == "deepfake_detection":
                    outputs = self.model(inputs).logits_labels.float()
                else:
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
                for inputs, labels, pathes in val_loader:  # val_loop:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    if self.model_name == "deepfake_detection":
                        outputs = self.model(inputs).logits_labels
                    else:
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
                print(
                    f"✅ New best model saved! Validation accuracy: {val_acc:.4f} (previous best: {best_val_acc:.4f})"
                )
                best_val_acc = val_acc

            # Also save a checkpoint every 5 epoch
            if epoch + 1 % 5 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved: {checkpoint_path}")

            print(
                f"Training:   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} ({correct}/{total})"
            )
            print(
                f"Validation: Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} ({val_correct}/{val_total})"
            )

            # Calculate and display improvement or regression
            if epoch > 0:
                train_loss_change = train_loss - train_losses[-2]
                train_acc_change = train_acc - train_accuracies[-2]
                val_loss_change = val_loss - val_losses[-2]
                val_acc_change = val_acc - val_accuracies[-2]

                print(f"Changes from previous epoch:")
                print(
                    f"  Train Loss: {train_loss_change:+.4f} | Train Acc: {train_acc_change:+.4f}"
                )
                print(
                    f"  Val Loss: {val_loss_change:+.4f} | Val Acc: {val_acc_change:+.4f}"
                )

            # Display current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr:.6f}")

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
                optimizer,
            )

        # Final summary at the end of training
        tqdm.write(f"\nTraining completed after {self.num_epochs} epochs")
        tqdm.write(f"Best validation accuracy: {best_val_acc:.4f}")

        # Load the best model before testing
        print(f"Loading best model from {model_save_path}")
        self.model.load_state_dict(torch.load(model_save_path))

        return self.model
