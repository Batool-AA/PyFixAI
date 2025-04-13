from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pytest
import jsonlines
import torch 

model_path = "../../model/codet5-fix-model"  # Path to your saved model folder

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def suggest_fix(code):
    model.eval()
    input_text = f"fix: {code}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=128)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_test_data():
    data = []
    with jsonlines.open("../sample_test_set.jsonl") as reader:
        for obj in reader:
            data.append((obj["src"], obj["tgt"]))
    return data

# Use parametrize correctly
@pytest.mark.parametrize("buggy_code,expected_fixed_code", load_test_data())
def test_model_predictions(buggy_code, expected_fixed_code):
    predicted_code = suggest_fix(buggy_code)
    assert predicted_code.strip() == expected_fixed_code.strip(), f"\nExpected:\n{expected_fixed_code}\nGot:\n{predicted_code}"
