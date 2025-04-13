from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import jsonlines
import torch 
import json 
import subprocess

model_path = "codebert-gpt2"  # Path to your saved model folder

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


def suggest_fix(code):
    model.eval()
    input_text = f"fix: {code}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=128)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def test_model():
    results = []
    scores = []
    with jsonlines.open("evaluation/sample_test_set.jsonl") as reader:
        for i, obj in enumerate(reader):
            fixed_code = suggest_fix(obj["src"])
            filename = f"generated_code.py"

            # Write fixed code to a file
            with open(filename, "w") as f:
                f.write(f"")
                f.write(f"# Fixed Code\n{fixed_code}\n")

            # Run pylint with parseable format
            pylint_result = subprocess.run(
                ["pylint", filename, "--output-format=parseable", "--exit-zero"],
                capture_output=True,
                text=True
            )

            # Extract score from the output
            pylint_output = pylint_result.stdout
            score = "N/A"

            # Find the final score line
            for line in pylint_output.split("\n"):
                if "Your code has been rated at" in line:
                    score = float(line.split(" ")[6].split("/")[0])  # Extract score from the output
                    scores.append(score)

            # Store results
            results.append({
                "test_case": obj["src"],
                "generated_code": fixed_code,
                "pylint_score": score
            })

    # Save all results to a JSON file
    with open("pylint_results.json", "w") as json_file:
        dump = ''
        json.dump(dump, json_file, indent=4)
        dump = {
                "avg_score": sum(scores)/len(scores),
                "results": results
               }
        json.dump(dump, json_file, indent=4)

    print("Pylint evaluation completed! Results saved in pylint_results.json")

test_model()