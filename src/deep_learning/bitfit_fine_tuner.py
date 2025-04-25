class BitfitFT(FullFT):
    def __init__(self, model_name, data_dir, real_folder, fake_folder, num_epochs, batch_size, learning_rate, use_wandb= False, model = None, processor = None):
        super().__init__(model_name, data_dir, real_folder, fake_folder, num_epochs, batch_size, learning_rate, use_wandb, model, processor)
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if ".bias" in name:
                param.requires_grad = True
        for name, param in self.model.classifier.named_parameters():
            param.requires_grad = True

    def Tune(self):
        """
        Args:
            data_dir (str): Directory containing 'train' and 'val' subdirectories,
                            each with 'real' and 'fake' subdirectories
            real_folder:
            fake_folder:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            use_wandb (boolean): Use wandb's logging
        """
        return super().Tune()
