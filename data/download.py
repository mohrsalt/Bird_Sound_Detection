import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import soundfile as sf
import datasets
from datasets import Audio


# Create output directory


XCM = datasets.load_dataset(
    name="XCM",
    path='DBD-research-group/BirdSet',
    trust_remote_code=True, 
   cache_dir='/home/users/ntu/bhargavi/scratch/Birdset'
)


