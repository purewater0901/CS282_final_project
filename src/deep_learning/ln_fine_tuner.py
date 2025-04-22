from deep_learning.full_fine_tuner import FullFT
from peft import LNTuningConfig, get_peft_model
from torch import optim


# Layer Normalization Fine Tuning (LNFT)

class LNFT(FullFT):
    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate=None,
        model=None,
        processor=None,
        device="cpu",
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
            device,
        )
        self.method_name = "Layer_Norm_FT"
        self.ln_config = LNTuningConfig(
            target_modules=kwargs.get("target_modules"),
        )

    def Tune(self):
        self.freeze_all_layers()
        
        # Unfreeze the linear classifier layer
        for param in self.model.model.parameters():
            param.requires_grad = True

        self.model = get_peft_model(self.model, self.ln_config)
        print("Layer Norm model loaded")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        model_save_path = f"ln_tune_{self.model_name}.pth"

        return super().Tune(optimizer=optimizer, model_save_path=model_save_path)