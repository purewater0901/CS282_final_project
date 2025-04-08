from deep_learning.full_fine_tuner import FullFT
from peft import LoraConfig, get_peft_model
from torch import optim


class LoraFT(FullFT):
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
        self.method_name = "Lora_FT"
        self.lora_config = LoraConfig(
            r=kwargs.get("r"),
            lora_alpha=kwargs.get("lora_alpha"),
            target_modules=kwargs.get("target_modules"),
            lora_dropout=kwargs.get("lora_dropout"),
            bias=kwargs.get("bias"),
        )

    def Tune(self):
        self.model = get_peft_model(self.model, self.lora_config)
        print("Lora model loaded")
        self.model.print_trainable_parameters()

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        model_save_path = f"lora_tune_{self.model_name}.pth"

        return super().Tune(optimizer=optimizer, model_save_path=model_save_path)
