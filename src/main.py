import os
import numpy as np
import torch
from transformers import CLIPProcessor
import argparse
import wandb

from deep_learning.full_fine_tuner import FullFT
from deep_learning.frozen_fine_tuner import FrozenFT
from deep_learning.linear_tail_fine_tuner import LinearTailFT
from deep_learning.lora_fine_tuner import LoraFT
from deep_learning.vera_fine_tuner import VeraFT
from deep_learning.bayes_tuner import BayesTune
from deep_learning.ln_fine_tuner import LNFT
from deep_learning.adapter_tuner import Adapter_FT
from deep_learning.bitfit_fine_tuner import BitfitFT
from utils.config_parser import parse_cfg

from deepfake_detection.model.dfdet import DeepfakeDetectionModel
from deepfake_detection.config import Config

TUNER_CLASSES = {
    "FrozenFT": FrozenFT,
    "FullFT": FullFT,
    "LinearTailFT": LinearTailFT,
    "LoraFT": LoraFT,
    "VeraFT": VeraFT,
    "BayesFT": BayesTune,
    "LNFT": LNFT,
    "Adapter_FT": Adapter_FT,
    "BitfitFT": BitfitFT,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    cfg = parser.parse_args()
    cfg = parse_cfg(cfg, cfg.config)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True # Added this
    torch.backends.cudnn.benchmark = False  # Change this to False
    torch.set_float32_matmul_precision("high")

    # sets model if necessary
    model = None
    preprocess = None
    if cfg.model_name == "deepfake_detection":
        model_path = "weights/model.ckpt"
        if not os.path.exists(model_path):
            print("Downloading model")
            os.makedirs("weights", exist_ok=True)
            os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.ckpt -O {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")

        model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
        model.load_state_dict(ckpt["state_dict"])

        # Get preprocessing function
        #preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast = True)
        preprocess = model.get_preprocessing()

    # create config
    cfg.data_dir = os.path.join(os.getcwd(), "data")  # Ensure the data directory is correct
    cfg.model = model if model is not None else None
    cfg.processor = preprocess if preprocess is not None else None

    # initialize wandb
    project_name = "Fine-Tuning Experiment"
    mode = "online" if cfg.use_wandb else "disabled"
    group_name = cfg.tuning_type
    run_name = f"{cfg.model_name}_{cfg.tuning_type}" # Update run name
    wandb.init(project=project_name, group=group_name, name=run_name, config=cfg, mode=mode)

    tuner_cls = TUNER_CLASSES.get(cfg.tuning_type)
    if tuner_cls is None:
        raise ValueError(f"Unsupported Tuning Type: {cfg.tuning_type}")
    tuner = tuner_cls(**vars(cfg))

    tuned_model = tuner.Experiment()
