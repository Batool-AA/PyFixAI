import torch 
from transformers import AutoModel
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CodeErrorFixModel(nn.Module):
    def __init__(self, encoder_model_name, vocab_size, embed_size=768, num_decoder_layers=6, nhead=8):
        super().__init__()
        # Load the pretrained CodeBERT encoder
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        # Decoder components
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size
    
    def generate_square_subsequent_mask(self, sz):
        # Create a mask to ensure that each position only attends to previous positions
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        # Encode source sequence
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask)
        memory = encoder_outputs.last_hidden_state  # shape: (batch_size, src_seq_len, embed_size)
        
        # Prepare target embeddings
        tgt_embeddings = self.decoder_embedding(tgt_input_ids) * math.sqrt(self.embed_size)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        # Transformer expects (seq_len, batch_size, embed_size)
        tgt_embeddings = tgt_embeddings.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        tgt_seq_len = tgt_input_ids.size(1)
        # Create target mask for auto-regressive generation
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        
        decoder_output = self.decoder(tgt=tgt_embeddings, memory=memory, tgt_mask=tgt_mask)
        # Transpose back: (batch_size, seq_len, embed_size)
        decoder_output = decoder_output.transpose(0, 1)
        logits = self.fc_out(decoder_output)  # (batch_size, seq_len, vocab_size)
        return logits

def encode_example(example, tokenizer, max_length=512):
    src_tokens = example['src']
    tgt_tokens = example['tgt']
    src_text = " ".join(src_tokens)
    
    # Add start and end tokens to the target
    tgt_text = "<s> " + " ".join(tgt_tokens) + " </s>"
    
    src_enc = tokenizer(src_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    tgt_enc = tokenizer(tgt_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    return src_enc, tgt_enc

def pre_tokenize_data(data, tokenizer, max_length=512):
    tokenized_data = []
    src_enc, tgt_enc = encode_example(data, tokenizer, max_length)
    tokenized_data.append({
        'src_input_ids': src_enc['input_ids'].squeeze(0),
        'src_attention_mask': src_enc['attention_mask'].squeeze(0),
        'tgt_input_ids': tgt_enc['input_ids'].squeeze(0),
        'tgt_attention_mask': tgt_enc['attention_mask'].squeeze(0)
    })
    return tokenized_data

import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, RobertaTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import pytest
import sys
from io import StringIO

# # Load the test dataset (expected target outputs)
# with open("../../dataset/data/python/processed_with_verdict/test.jsonl", "r") as file:
#     test_data = [json.loads(line) for i, line in enumerate(file) if i < 10]

# Load the test dataset (expected target outputs)

# test_data = test_data[:1]


# Token to code converter
def detokenize(tokens):
    code = ""
    indent_level = 0
    for token in tokens:
        if token == "NEW_LINE":
            code += "\n" + ("    " * indent_level)
        elif token == "INDENT":
            indent_level += 1
            code += "\n" + ("    " * indent_level)
        elif token == "DEDENT":
            indent_level -= 1
            code += "\n" + ("    " * indent_level)
        else:
            if code and code[-1] not in ("\n", " ", "(", "[", "{"):
                code += " "
            code += token
    return code.strip()

# Code runner
def run_python_code(code: str, input_data: str):
    old_stdout, old_stdin = sys.stdout, sys.stdin
    sys.stdout = output = StringIO()
    sys.stdin = StringIO(input_data)
    try:
        exec(code, {})
        return output.getvalue().strip()
    except Exception as e:
        return f"__ERROR__::{str(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin
        

def check_model_code_matches_target(model_tokens, target_tokens, input_data, decode=False):
    if decode:
        model_code = detokenize(model_tokens)
    else:
        model_code = model_tokens

    target_code = detokenize(target_tokens)
    model_output = run_python_code(model_code, input_data)
    target_output = run_python_code(target_code, input_data)

    assert model_output == target_output, f"Model: {model_output} | Target: {target_output}"


def evaluate_seq_2_seq(entry, tokenizer, model):
    src_code = entry["src"]
    expected_output = entry["tgt"]
    
    inputs = tokenizer(src_code, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        output = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128)

    predicted_code = tokenizer.decode(output[0], skip_special_tokens=True)
    check_model_code_matches_target(predicted_code, expected_output, entry["input"], decode=False)


def evaluate_custom(entry, model, tokenizer, device='cpu'):
    model.eval()
    model.to(device)

    predictions = []
    ground_truths = []

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
    predictions.append(predicted_text)
    ground_truths.append(target_text)

    check_model_code_matches_target(predicted_text, target_text, "3", decode=True)
