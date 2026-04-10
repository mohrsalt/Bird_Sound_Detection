import torch
import logging
from javad.src.javad.extras import Trainer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = Trainer(
        run_name="balanced_test",
        dataset_settings={
            #"audio_root": "/home/users/ntu/bhargavi/scratch/Birdset/downloads/extracted",
            "audio_root": "/home/users/ntu/bhargavi/scratch/train_dataset_top20/train",
	    "spectrograms_root": "/home/users/ntu/bhargavi/scratch/datasets_spec",
            "index_root": "/home/users/ntu/bhargavi/scratch/dataset_index",
            "metadata_json_path": "/home/users/ntu/bhargavi/scratch/train_dataset_top20/metadata_train_A.json",
            "max_memory_cache": 16000, # allow to use up to 16Gb of RAM to retain spectrograms 
        },
        use_mixed_precision=True,
        use_scheduler=True,
        window_min_content_ratio=0.25,
        window_offset_sec=0.5,
        device=torch.device("cuda:0"),
        learning_rate=1e-4,
	batch_size=16,
        num_workers=2,
        total_epochs=1,
        augmentations={
            "mix_chance": 0.5,
            "erase_chance": 0.5,
            "zero_chance": 0.01,
        },
    )
    trainer.train()
