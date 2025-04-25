from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import functions
from functions import pre_tokenize_data, cleanup_tokens, detokenize, tokenize_code
import sys
from dotenv import load_dotenv
import os
import torch
import torch.nn.functional as F

load_dotenv(dotenv_path="../.env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from frontend)

sys.modules['__main__'].CodeErrorFixModel = functions.CodeErrorFixModel
sys.modules['__main__'].PositionalEncoding = functions.PositionalEncoding
model_path = '../codebert-custom0/full_model.pth'
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = torch.load(model_path, map_location="cpu", weights_only=False)
device = "cpu"

if model:
    model.eval()  # Set model to evaluation mode
    model.to("cpu")
else:
    print("Failed to load the model, skipping inference.")






def suggest_fix(code: str):
    # print("input code", code)
    entry = tokenize_code(code)
    raw_tokens = tokenizer(entry, return_tensors="pt", padding=True)
    input_token_count = raw_tokens["input_ids"].shape[1]
    max_length = min(input_token_count * 8, 256)    
    # print("tokenized", entry)

    tokenized = pre_tokenize_data(entry, tokenizer, max_length=512)[0]
    src_input_ids = tokenized["src_input_ids"].unsqueeze(0).to(device)
    src_attention_mask = tokenized["src_attention_mask"].unsqueeze(0).to(device)

    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.pad_token_id

    tgt_ids = torch.tensor([[bos_id]], device=device)
    # input_len = src_input_ids.shape[1]  # number of input tokens
    # max_length = 4 * input_len  # cap at 256, or adjust this cap as needed
    # print("max:", max_length)
    generated_ids = tgt_ids

    repetition_count = 0
    last_token = None

    for step in range(max_length):
        tgt_att_mask = torch.ones_like(generated_ids, dtype=torch.bool, device=device)

        logits = model(src_input_ids, src_attention_mask, generated_ids, tgt_att_mask)
        last_logits = logits[:, -1, :]

        last_logits[0, bos_id] = -float("inf")
        last_logits[0, pad_id] = -float("inf")
        if step == 0:
            last_logits[0, eos_id] = -float("inf")

        next_id = torch.argmax(last_logits, dim=-1)

        if next_id.item() == eos_id:
            break

        if last_token is not None and next_id.item() == last_token:
            repetition_count += 1
        else:
            repetition_count = 0
        last_token = next_id.item()

        if repetition_count >= 3:
            print("Detected repetition. Breaking early.")
            break

        generated_ids = torch.cat([generated_ids, next_id.unsqueeze(0)], dim=1)

    output_ids = generated_ids[0, 1:]
    fixed_code = tokenizer.decode(output_ids, skip_special_tokens=True)

    # # Remove artifacts
    # fixed_code = fixed_code.replace("▁", " ").strip()

    # # Truncate after first double newline (start of unrelated code)
    # if "\n\n" in fixed_code:
    #     fixed_code = fixed_code.split("\n\n")[0]

    # print("raw prediction:", fixed_code)
    print("output", fixed_code)
    

    predicted_text = cleanup_tokens(fixed_code)
    print("cleaned:", predicted_text)

    detok = detokenize(predicted_text.split())
    print("detokenized:", detok)
    return detok






def explain_code(code: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains Python code."},
                {"role": "user", "content": f"Explain this Python code:\n{code}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error explaining code: {e}"
    
@app.route("/suggest_fix", methods=["POST"])
def suggest_fix_api():
    data = request.get_json()
    code = data.get("buggy_code", "")
    print(code)
    if not code:
        return jsonify({"error": "buggy_code is required"}), 400

    fix = suggest_fix(code)
    return jsonify({"fixed_code": fix})

@app.route("/explain_code", methods=["POST"])
def explain_code_api():
    data = request.get_json()
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "code is required"}), 400

    explain= explain_code(code)
    return jsonify({"explanation": explain})

if __name__ == "__main__":
    app.run(debug=True)