from transformers import AutoModel
import torch.nn as nn
import math
import io
import tokenize
import re
import torch

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
    # src_tokens = example['src']
    # tgt_tokens = example['tgt']
    src_tokens = example
    src_text = " ".join(src_tokens)
    
    # Add start and end tokens to the target
    # tgt_text = "<s> " + " ".join(tgt_tokens) + " </s>"
    
    src_enc = tokenizer(src_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    # tgt_enc = tokenizer(tgt_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    return src_enc #, tgt_enc

def pre_tokenize_data(data, tokenizer, max_length=512):
    tokenized_data = []
    # src_enc, tgt_enc = encode_example(data, tokenizer, max_length)
    src_enc = encode_example(data, tokenizer, max_length)
    print("src_enc", src_enc)
    tokenized_data.append({
        'src_input_ids': src_enc['input_ids'].squeeze(0),
        'src_attention_mask': src_enc['attention_mask'].squeeze(0),
        # 'tgt_input_ids': tgt_enc['input_ids'].squeeze(0),
        # 'tgt_attention_mask': tgt_enc['attention_mask'].squeeze(0)
    })
    return tokenized_data

def cleanup_tokens(token_str):
    import re

    # Normalize whitespace
    token_str = token_str.replace('\n', ' ').replace('\r', ' ')
    token_str = re.sub(r'\s+', ' ', token_str).strip()

    # Tokenize string
    quoted_tokens = re.findall(r'"[^"]*"|\S+', token_str)
    cleaned = []

    valid_symbols = {
        "NEW_LINE", "INDENT", "DEDENT",
        "(", ")", "[", "]", "{", "}", ":", ",", ".", "+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">=", ":=", '"'
    }

    valid_keywords = {"NEW_LINE", "INDENT", "DEDENT"}
    garbage_start_index = None

    # Drop garbage after patterns like = = = = =
    for i, token in enumerate(quoted_tokens):
        if token in {"=", "_", "+", "-", "[", "]"} and i + 3 < len(quoted_tokens):
            pattern = quoted_tokens[i:i+5]
            if all(p == token for p in pattern):
                garbage_start_index = i
                break

    if garbage_start_index is not None:
        quoted_tokens = quoted_tokens[:garbage_start_index]

    # Remove incomplete structured token at end (like 'NEW', 'IND')
    if quoted_tokens:
        last = quoted_tokens[-1]
        if any(last != kw and kw.startswith(last) for kw in valid_keywords):
            quoted_tokens = quoted_tokens[:-1]

    for token in quoted_tokens:
        if token in valid_symbols:
            cleaned.append(token)
        elif token in valid_keywords:
            cleaned.append(token)
        elif re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", token):
            cleaned.append(token)
        elif re.fullmatch(r"\d+", token):
            cleaned.append(token)
        elif token.startswith('"') and token.endswith('"'):
            cleaned.append(token)

    return " ".join(cleaned)

def detokenize(tokens):
    code = []
    indent_level = 0
    i = 0

    no_space_before = {')', ']', '}', '.', ',', ':', '(', '[', '{'}
    no_space_after = {'(', '[', '{', '.'}

    at_line_start = True

    while i < len(tokens):
        token = tokens[i]

        if token == 'NEW_LINE':
            code.append('\n')
            at_line_start = True

        elif token == 'INDENT':
            indent_level += 1

        elif token == 'DEDENT':
            indent_level = max(indent_level - 1, 0)

        else:
            if at_line_start:
                # Only indent if we're not at the very start
                if code:
                    code.append('    ' * indent_level)
                at_line_start = False

            if code and not code[-1].endswith('\n') and \
               token not in no_space_before and \
               (code[-1] and code[-1][-1] not in no_space_after):
                code.append(' ')

            code.append(token)

        i += 1

    return ''.join(code)

def tokenize_code(code):
    lines = code.strip().split('\n')
    tokens = []

    for line in lines:
        line = line.rstrip()
        if line == "":
            continue

        indent_level = (len(line) - len(line.lstrip(' '))) // 4
        if indent_level > 0:
            tokens.extend(["INDENT"] * indent_level)

        parts = re.findall(r'\w+|[^\s\w]', line)
        tokens.extend(parts)
        tokens.append("NEW_LINE")

    return [t for t in tokens if t]

