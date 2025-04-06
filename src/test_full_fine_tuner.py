import os
from deep_learning.full_fine_tuner import FullFT

if __name__ == "__main__":
    # Load Vanilla Model
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    config = {
        "model_name": "efficientnetv2_rw_m.agc_in1k",  #'swin_large_patch4_window7_224.ms_in22k',#'vit_base_patch16_clip_224.openai', #'swinv2_cr_tiny_ns_224.sw_in1k', #'xception', #'vit_base_patch16_clip_224.openai',
        "data_dir": data_dir,
        "real_folder": "Real_split",
        "fake_folder": "StyleGAN_split",
        "num_epochs": 1,
        "batch_size": 16,
        "learning_rate": 0.00001,
    }

    FullFineTuner = FullFT(**config)
    FullFineTuner.set_TestFolder('firefly_split')
    tuned_model = FullFineTuner.Experiment('swin_large_C1')