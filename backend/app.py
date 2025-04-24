from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
from openai import OpenAI
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import functions
from functions import pre_tokenize_data, cleanup_tokens, detokenize, tokenize_code
import sys

# Initialize app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from frontend)

sys.modules['__main__'].CodeErrorFixModel = functions.CodeErrorFixModel
sys.modules['__main__'].PositionalEncoding = functions.PositionalEncoding
model_path = "C:/Users/User/Documents/GitHub/PyFixAI/codebert-custom0/full_model.pth"
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = torch.load(model_path, map_location="cpu", weights_only=False)
device = 'cpu'

client = OpenAI(api_key="" )

def suggest_fix(code: str):
    print("input code", code)
    entry = tokenize_code(code)
    print("tokenized", entry)
    tokenized_test_data = pre_tokenize_data(entry, tokenizer, max_length=512)
    test_loader = DataLoader(tokenized_test_data, batch_size=1)
    
    with torch.no_grad():
        for batch in test_loader:
            src_input_ids = batch['src_input_ids'].to(device)
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)

            # 🔮 Predict (use teacher forcing)
            output = model(src_input_ids, src_attention_mask, tgt_input_ids[:, :-1], tgt_attention_mask[:, :-1])
            predicted_ids = output.argmax(dim=-1)

            # Decode Input (buggy), Prediction, and Target (ground truth)
            input_text = tokenizer.decode(src_input_ids[0], skip_special_tokens=True)
            predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            target_text = tokenizer.decode(tgt_input_ids[0], skip_special_tokens=True)
    
    # print("output - before clean\n", predicted_text)
    predicted_text = cleanup_tokens(predicted_text)
    # print("output after clean\n", predicted_text)
    predicted_text = detokenize(predicted_text.split())
    return predicted_text

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