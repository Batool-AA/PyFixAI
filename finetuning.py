import json

def load_data_chunk(file_path, max_lines=1000):
    """Loads a limited number of lines from a large JSONL dataset to avoid memory issues."""
    data = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break  # Stop after loading max_lines
            entry = json.loads(line)
            buggy_code = " ".join(entry["src"]).replace(" NEW_LINE ", "\n").replace(" INDENT ", "    ").replace(" DEDENT ", "")
            fixed_code = " ".join(entry["tgt"]).replace(" NEW_LINE ", "\n").replace(" INDENT ", "    ").replace(" DEDENT ", "")
            data.append((buggy_code, fixed_code))
    return data


train_data_sample = load_data_chunk("dataset/data/python/processed/train.jsonl", max_lines=10)

# Check the first example
print(train_data_sample[0])
