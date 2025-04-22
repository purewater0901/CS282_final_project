from deep_learning.full_fine_tuner import FullFT
from peft import VeraConfig, get_peft_model
from torch import optim


class VeraFT(FullFT):
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
        self.method_name = "Vera_FT"
        self.vera_config = VeraConfig(
            r=kwargs.get("r"),
            target_modules=kwargs.get("target_modules"),
            vera_dropout=kwargs.get("vera_dropout"),
            bias=kwargs.get("bias"),
        )

    def Tune(self):
        # Freeze all layers
        self.freeze_all_layers()

        self.model = get_peft_model(self.model, self.vera_config)
        print("Vera model loaded")
        #self.model.print_trainable_parameters()

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        model_save_path = f"vera_tune_{self.model_name}.pth"

        return super().Tune(optimizer=optimizer, model_save_path=model_save_path)