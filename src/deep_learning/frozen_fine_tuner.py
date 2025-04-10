import os
from tqdm.auto import tqdm
import torch
from torch import nn, optim

from deep_learning.fine_tuner import FineTuner


class FrozenFT(FineTuner):
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
        self.method_name = "Frozen_FT"

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.print_trainable_parameters()

    def Tune(self):
        return self.model