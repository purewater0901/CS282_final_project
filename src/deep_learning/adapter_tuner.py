from deep_learning.full_fine_tuner import FullFT
from torch import optim, nn



class AdapterWrapper(nn.Module):
    def __init__(self, base_model, hidden_dim=1024, bottleneck=1024*2, dropout=0.1):
        super().__init__()

        """
        Tried nn.Sequential(*list(base_model.children())[:-5]) as well but didn't work. 
        It's better to directly refer the feature extractor/model.

        Also, no need to keep the base_model in the wrapper, we can just use the feature extractor and the head.
        """
        self.feature_extracter = base_model.feature_extractor 
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU() # Added this nonlinearity otherwise nn.Linear doesn't mean anything
        )
        self.original_head = base_model.model


    def forward(self, x):
        features = self.feature_extracter(x) # Extract features    
        adapted = features + self.adapter(features)  # residual connection
        headoutput = self.original_head(adapted)
        
        return headoutput

    
    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Transplanted from the peft liblary
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )


class Adapter_FT(FullFT):
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
        self.method_name = "Adapter_FT"
        self.hidden_dim = kwargs.get("hidden_dim")
        self.bottleneck = kwargs.get("bottleneck")
        self.dropout = kwargs.get("dropout")

    def Tune(self):
        # Freeze all layers
        self.freeze_all_layers()

        # Add a 2-layer MLP to the original model
        self.model = AdapterWrapper(self.model, self.hidden_dim, self.bottleneck, self.dropout).to(self.device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        model_save_path = f"adapter_tune_{self.model_name}.pth"

        return super().Tune(optimizer=optimizer, model_save_path=model_save_path)
