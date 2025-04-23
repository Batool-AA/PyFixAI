from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
from flask_cors import CORS
import openai
from openai import OpenAI
import os

# Initialize app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from frontend)

# Load tokenizer and model
# model_path = "../model/codebert-bug-type"
# tokenizer = RobertaTokenizer.from_pretrained(model_path)
# model = RobertaForSequenceClassification.from_pretrained(model_path)
# model.eval()

# label_map = {
#     0: "clean",
#     1: "missing_colon",
#     2: "indentation_error",
#     3: "undefined_variable",
#     4: "unmatched_brackets",
#     5: "index_error"
# }

# @app.route("/predict", methods=["POST"])
# def predict_bug_type():
#     try:
#         data = request.json
#         code = data.get("code", "")
#         if not code:
#             return jsonify({"error": "No code provided"}), 400

#         inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=128)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             predicted_id = torch.argmax(outputs.logits, dim=-1).item()
#             bug_type = label_map[predicted_id]

#         return jsonify({"bug_type": bug_type})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

model_path = "Salesforce/codet5-base" 
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def suggest_fix(code: str):
    # Convert input to token format used in training
    tokenized_input = " ".join(code.replace("\n", " NEW_LINE ").replace("    ", " INDENT ").split())
    input_text = f"fix: {tokenized_input}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(" NEW_LINE ", "\n").replace(" INDENT ", "    ")


# Make sure to set your API key
client = OpenAI(api_key="" ) # or directly set: openai.api_key = "your-key"

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