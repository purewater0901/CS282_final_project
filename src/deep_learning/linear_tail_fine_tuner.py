from deep_learning.full_fine_tuner import FullFT
from torch import optim

class LinearTailFT(FullFT):
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
        self.method_name = "LinearTail_FT"

    def Tune(self):
        # freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # enable training for the last layer
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.model.get_classifier().parameters(), lr=self.learning_rate)
        model_save_path = f"linear_tail_tune_{self.model_name}.pth"

        return super().Tune(optimizer=optimizer, model_save_path=model_save_path)