import json

train_file = "dataset/data/python/processed_with_verdict/train.jsonl"

# Read only a small part of the file
with open(train_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load entire JSON list

# Check the first example
sample = data[0]  # First entry
print(sample)
sample1 = data[1] 
print(sample1)