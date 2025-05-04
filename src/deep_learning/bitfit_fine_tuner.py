from deep_learning.full_fine_tuner import FullFT

class BitfitFT(FullFT):
    def __init__(
        self, 
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate = None,
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
        self.method_name = "Bitfit_FT"

    def Tune(self):
        self.freeze_all_layers()

        for name, param in self.model.named_parameters():
            if ".bias" in name:
                param.requires_grad = True

        # Unfreeze the linear classifier layer
        for param in self.model.model.parameters():
            param.requires_grad = True

        model_save_path = f"bitfit_tune_{self.model_name}.pth"
        return super().Tune(model_save_path=model_save_path)