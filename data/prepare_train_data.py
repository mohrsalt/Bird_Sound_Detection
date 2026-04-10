import json
import os
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from collections import defaultdict


# Input: Pure class A list generated in the previous stage
INPUT_JSON = "/home/users/ntu/bhargavi/scratch/pure_bird_manifests/train_A.json"
# Output: Final dataset directory
OUTPUT_ROOT = "/home/users/ntu/bhargavi/scratch/train_dataset_top20" # Modify output path to avoid overwriting

# Filtering configuration
TOP_N = 20  # Only select Top 20 bird species

# Generation target configuration
TARGET_TRAIN_HOURS = 150.0  # Train set target: 150 hours
TARGET_DEV_HOURS = 50.0     # Dev set target: 50 hours

# Single file generation configuration
MIN_SESSION_DURATION = 5 * 60  
SR = 32000
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4

# 静音间隔 (秒)
SILENCE_MIN = 0.5
SILENCE_MAX = 1.0

SEED = 42

def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        return y
    except:
        return None

def filter_top_n_species(data_list, n=20):

    print("🔍 Counting and filtering Top 20 bird species...")

    # Calculate durations
    species_durations = defaultdict(float)
    species_clips = defaultdict(list)

    for item in data_list:
        code = item.get('ebird_code')
        if not code: continue

        species_clips[code].append(item)

        # Calculate effective duration of the clip
        dur = 0.0
        if item.get('detected_events'):
            for e in item['detected_events']:
                dur += (e[1] - e[0])
        else:
            dur = 10.0
        species_durations[code] += dur

    # Sort
    sorted_species = sorted(species_durations.items(), key=lambda x: x[1], reverse=True)
    top_n_species = [x[0] for x in sorted_species[:n]]

    print(f"🏆 Top {n} species list: {top_n_species}")

    # Filter clips
    filtered_data = []
    for sp in top_n_species:
        filtered_data.extend(species_clips[sp])

    print(f"✅ Filtering completed: Original clips reduced from {len(data_list)} to {len(filtered_data)} (only Top {n})")
    return filtered_data, top_n_species

class LargeScaleGenerator:
    def __init__(self, clip_pool, species_list):
        self.clip_pool = clip_pool
        self.all_species = list(species_list) # Top 20
        self.sp_to_clips = defaultdict(list)
        for c in clip_pool:
            self.sp_to_clips[c['ebird_code']].append(c)
        self.metalist={}

    def get_random_clip(self, species):
        if not self.sp_to_clips[species]: return None
        return random.choice(self.sp_to_clips[species])

    def generate_session(self, session_idx, output_dir, split_name):
        """Generate a single non-overlapping file >=20min"""

        # 1. Randomly decide the number of speakers in this session (2-4)
        num_speakers = random.randint(MIN_SPEAKERS, MAX_SPEAKERS)

        # 2. Randomly select speakers (from Top 20)
        active_species = random.sample(self.all_species, num_speakers)

        # 3. Generate alternating speech
        full_audio_buffer = []
        session_events = []
        current_time = 0.0
        speech=[]
        # Loop until the duration requirement is met
        while current_time < MIN_SESSION_DURATION:
            # Randomly select a bird currently speaking
            current_spk = random.choice(active_species)

            # Get clip
            clip = self.get_random_clip(current_spk)
            if not clip: continue

            y = load_audio(clip['audio_path'])
            if y is None: continue

            # Generate silence (key to no overlap)
            silence_sec = random.uniform(SILENCE_MIN, SILENCE_MAX)
            silence_samples = int(silence_sec * SR)
            silence_arr = np.zeros(silence_samples, dtype=np.float32)

            # Append to buffer
            full_audio_buffer.append(silence_arr)
            full_audio_buffer.append(y)

            # Calculate timestamps
            audio_start = current_time + silence_sec
            audio_dur = len(y) / SR

            # Process event labels
            if clip.get('detected_events'):
                for e in clip['detected_events']:
                    session_events.append({
                        "label": current_spk,
                        "start": float(f"{audio_start + e[0]:.4f}"),
                        "end": float(f"{audio_start + e[1]:.4f}"),
                        "original_file": os.path.basename(clip['audio_path'])
                    })
                    speech.append([int((audio_start + e[0])*32000),int((audio_start + e[1])*32000)]);

            else:
                session_events.append({
                    "label": current_spk,
                    "start": float(f"{audio_start:.4f}"),
                    "end": float(f"{audio_start + audio_dur:.4f}"),
                    "original_file": os.path.basename(clip['audio_path'])
                })
                speech.append([int((audio_start)*32000),int((audio_start + audio_dur)*32000)]);

            current_time += (silence_sec + audio_dur)

        # 4. SAVE
        if not full_audio_buffer: return 0.0

        final_audio = np.concatenate(full_audio_buffer)
        final_dur = len(final_audio) / SR

        filename = f"{split_name}_top20_{session_idx:05d}_{num_speakers}spk.wav"
        out_wav = os.path.join(output_dir, filename)
        sf.write(out_wav, final_audio, SR)

        meta = {
            "filename": filename,
            "duration_sec": final_dur,
            "num_speakers": num_speakers,
            "speaker_list": active_species,
            "events": sorted(session_events, key=lambda x: x['start'])
        }
        self.metalist[filename]={
            "length":int(final_dur*32000),
             "intervals":{
                 "speech":sorted(speech, key=lambda x: x[0])
             },
             "metadata":{}
            }

        out_json = os.path.join(output_dir, filename.replace('.wav', '.json'))
        with open(out_json, 'w') as f:
            json.dump(meta, f, indent=2)

        return final_dur

def generate_dataset_partition(split_name, clip_pool, species_list, target_hours, output_root):
    out_dir = os.path.join(output_root, split_name)
    os.makedirs(out_dir, exist_ok=True)

    gen = LargeScaleGenerator(clip_pool, species_list)

    current_seconds = 0.0
    target_seconds = target_hours * 3600
    idx = 0

    print(f"\n🚀 Starting generation of {split_name} dataset...")
    print(f"   🎯 Target duration: {target_hours} hours ({target_seconds} seconds)")
    print(f"   ℹ️  Using clip pool size: {len(clip_pool)} clips (fully shared)")


    pbar = tqdm(total=int(target_seconds), desc=f"Generating {split_name}", unit="sec")

    while current_seconds < target_seconds:
        idx += 1
        dur = gen.generate_session(idx, out_dir, split_name)

        current_seconds += dur
        pbar.update(int(dur))

    pbar.close()
    out_json = os.path.join(output_root, "metadata_train_A_2.json")
    with open(out_json, 'w') as f:
            json.dump(gen.metalist, f, indent=2)
    print(f"✅ {split_name} completed! Actual total duration: {current_seconds/3600:.2f} hours")

def main():
    random.seed(SEED)

    if not os.path.exists(INPUT_JSON):
        print(f"❌ Input file does not exist: {INPUT_JSON}")
        return

    # 1. Read raw list
    with open(INPUT_JSON, 'r') as f:
        raw_data = json.load(f)


    filtered_data, top_20_species = filter_top_n_species(raw_data, n=TOP_N)

    shared_pool = filtered_data

    print("\n" + "="*60)
    print(f"   - Shared clip pool size: {len(shared_pool)} clips")
    print("="*60)

    # 4. Generate Train dataset (target 150h)
    generate_dataset_partition("train", shared_pool, top_20_species, TARGET_TRAIN_HOURS, OUTPUT_ROOT)

    # 5. Generate Dev dataset (target 20h)
    # Note: shared_pool is also passed here
    #generate_dataset_partition("dev", shared_pool, top_20_species, TARGET_DEV_HOURS, OUTPUT_ROOT)

    # print("\n" + "="*60)
    print(f"All generation completed! Output directory: {OUTPUT_ROOT}")
    # print("✅ Conditions met:")
    # print("   1. Only Top 20 bird species")
    # print("   2. Single file duration >= 20min")
    # print("   3. Number of speakers 2-4")
    # print("   4. No overlap (0-Overlap)")
    # print(f"   5. Train total duration >= {TARGET_TRAIN_HOURS}h")
    # print(f"   6. Dev total duration >= {TARGET_DEV_HOURS}h")
    # print("   7. **Note**: Train and Dev use the same source clip pool (fully shared)")
    # print("="*60)

if __name__ == "__main__":
    main()
