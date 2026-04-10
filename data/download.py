import datasets
import json
import os
from tqdm import tqdm


CACHE_DIR = '/workspaces/DATA/bird_data/BIRD_DIA'
DATASET_PATH = 'DBD-research-group/BirdSet'
DATASET_CONFIG = "XCM"
SPLIT = "train"
OUTPUT_DIR = "./pure_bird_manifests"

def load_dataset_metadata(cache_dir, path, config, split):

    try:
        dataset = datasets.load_dataset(
            path=path, name=config, split=split,
            trust_remote_code=True, cache_dir=cache_dir
        )
        return dataset
    except Exception as e:
        print(f"Error: {e}")
        return None

def is_pure_sample(sample):

    sec = sample.get('ebird_code_secondary')

    if sec is None or len(sec) == 0:
        return True
    return False

def calculate_event_duration(events):

    dur = 0.0
    if events:
        for evt in events:
            if len(evt) >= 2 and evt[1] > evt[0]:
                dur += (evt[1] - evt[0])
    return dur

def extract_metadata(sample):

    return {
        'audio_path': sample['audio']['path'],
        'ebird_code': sample['ebird_code'],
        'quality': sample['quality'],
        'detected_events': sample['detected_events'],
        'sr': 32000
    }

def process_and_save_pure_files(dataset, output_dir):

    pure_data = {
        'A': [],
        'B': [],
        'C': []
    }

    stats_duration = {'A': 0.0, 'B': 0.0, 'C': 0.0}


    for sample in tqdm(dataset, desc="Filtering"):
        quality = sample.get('quality')

        if quality in pure_data:

            if is_pure_sample(sample):

                events = sample.get('detected_events')
                duration = calculate_event_duration(events)

                if duration > 0:
                    stats_duration[quality] += duration


                    metadata = extract_metadata(sample)
                    pure_data[quality].append(metadata)


    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print(f"{'Quality':<10} | {'Pure Files':<12} | {'Pure Duration (hrs)':<20} | {'Saved To'}")
    print("-" * 80)

    for q in ['A', 'B', 'C']:

        output_filename = f"pure_class_{q}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pure_data[q], f, indent=2)


        count = len(pure_data[q])
        hours = stats_duration[q] / 3600

        print(f"{q:<10} | {count:<12} | {hours:<20.2f} | {output_filename}")

    print("-" * 80)
    print(f"The results are saved as: {output_dir}")
    print("💡 Tip: pure_class_A.json is the best material library for building the Overlap dataset.")

def main():
    dataset = load_dataset_metadata(CACHE_DIR, DATASET_PATH, DATASET_CONFIG, SPLIT)
    if dataset:
        process_and_save_pure_files(dataset, OUTPUT_DIR)

if __name__ == "__main__":
    main()
