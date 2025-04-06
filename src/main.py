import os
import torch
from transformers import CLIPProcessor
import argparse
import wandb

from deep_learning.full_fine_tuner import FullFT
from deep_learning.frozen_fine_tuner import FrozenFT
from utils.config_parser import parse_cfg

torch.set_float32_matmul_precision("high")

TUNER_CLASSES = {
    "FrozenFT": FrozenFT,
    "FullFT": FullFT,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", default="cuda:0", type=str)
    cfg = parser.parse_args()
    cfg = parse_cfg(cfg, cfg.config)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # sets model if necessary
    model = None
    preprocess = None
    if cfg.model_name == "deep_fake_detection":
        # Check if weights/model.torchscript exists, if not, download it from huggingface
        model_file = "weights/model.torchscript"
        model_path = os.path.join(os.getcwd(), model_file)
        if not os.path.exists(model_path):
            print("Downloading model")
            os.makedirs("weights", exist_ok=True)
            os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.torchscript -O {model_path}")

        # Load checkpoint
        model = torch.jit.load(model_path, map_location=cfg.device)

        # Load preprocessing function
        preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # create config
    config = {
        "model_name": cfg.model_name,
        "data_dir": os.path.join(os.getcwd(), "data"),  # Ensure the data directory is correct
        "real_folder": "Real_split",
        "fake_folder": "StyleGAN_split",
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "model": model if model is not None else None,
        "processor": preprocess if preprocess is not None else None,
    }

    # initialize wandb
    project_name = "Fine-Tuning Experiment"
    mode = "online" if cfg.use_wandb else "disabled"
    group_name = cfg.tuning_type
    run_name = cfg.model_name
    wandb.init(project=project_name, group=group_name, name=run_name, config=config, mode=mode)

    tuner_cls = TUNER_CLASSES.get(cfg.tuning_type)
    if tuner_cls is None:
        raise ValueError(f"Unsupported Tuning Type: {cfg.tuning_type}")
    tuner = tuner_cls(**config)

    tuner.set_TestFolder('firefly_split')
    tuned_model = tuner.Experiment()