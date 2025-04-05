import os
import torch
from PIL import Image
from transformers import CLIPProcessor

from deep_learning.frozen_fine_tuner import FrozenFT

DEVICE = "cuda:0"

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # Check if weights/model.torchscript exists, if not, download it from huggingface
    model_file = "weights/model.torchscript"
    model_path = os.path.join(os.getcwd(), model_file)
    if not os.path.exists(model_path):
        print("Downloading model")
        os.makedirs("weights", exist_ok=True)
        os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.torchscript -O {model_path}")

    # Load checkpoint
    model = torch.jit.load(model_path, map_location=DEVICE)

    # Load preprocessing function
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # create config
    config = {
        "model_name": "deep_fake_detection",  #'swin_large_patch4_window7_224.ms_in22k',#'vit_base_patch16_clip_224.openai', #'swinv2_cr_tiny_ns_224.sw_in1k', #'xception', #'vit_base_patch16_clip_224.openai',
        "data_dir": os.path.join(os.getcwd(), "data"),  # Ensure the data directory is correct
        "real_folder": "Real_split",
        "fake_folder": "StyleGAN_split",
        "num_epochs": 1,
        "batch_size": 16,
        "learning_rate": None,
        "use_wandb": False,
        "model": model,
        "processor": preprocess,
    }

    FullFineTuner = FrozenFT(**config)
    FullFineTuner.set_TestFolder('firefly_split')
    tuned_model = FullFineTuner.Experiment(wandb_run_name='swin_large_C1')