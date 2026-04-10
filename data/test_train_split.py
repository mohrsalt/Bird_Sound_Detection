import json
import random
with open('/home/users/ntu/bhargavi/scratch/pure_bird_manifests/pure_class_A.json', 'r') as file:
        data = json.load(file)

train_file = "/home/users/ntu/bhargavi/scratch/pure_bird_manifests/train_A.json"
test_file = "/home/users/ntu/bhargavi/scratch/pure_bird_manifests/test_A.json"

random.shuffle(data)

# Compute split index (80/20)
split_index = int(0.8 * len(data))

train_data = data[:split_index]
test_data = data[split_index:]

# Save the split lists
with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)
