from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, RobertaTokenizer
import pytest
import torch
from py_test_10 import evaluate_custom, evaluate_seq_2_seq
import py_test_10
import sys
sys.modules['__main__'].CodeErrorFixModel = py_test_10.CodeErrorFixModel
sys.modules['__main__'].PositionalEncoding = py_test_10.PositionalEncoding

import json

with open("../sample_test_set.jsonl", "r") as file:
    test_data = [json.loads(line) for i, line in enumerate(file)]

@pytest.fixture(scope="module")
def codeT5():
    model_path = "Salesforce/codeT5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# @pytest.fixture(scope="module")
# def codeBERTGPT2():
#     model_path = "../../codebert-gpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     return tokenizer, model

# @pytest.fixture(scope="module")
# def codeBERTcodeBERT():
#     model_path = "../../codebert-codebert"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     return tokenizer, model

# @pytest.fixture(scope="module")
# def codeBERTcustom0():
#     model_path = "../../codebert-custom0/full_model.pth"
#     tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#     model = torch.load(model_path, map_location="cpu", weights_only=False)
#     return tokenizer, model

# @pytest.fixture(scope="module")
# def codeBERTcustom1():
#     model_path = "../../codebert-custom1/full_model.pth"
#     tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#     model = torch.load(model_path, map_location="cpu", weights_only=False)
#     return tokenizer, model

# @pytest.fixture(scope="module")
# def codeBERTcustom2():
#     model_path = "../../codebert-custom2/full_model.pth"
#     tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#     model = torch.load(model_path, map_location="cpu", weights_only=False)
#     return tokenizer, model

# @pytest.fixture(scope="module")
# def codeBERTcustom3():
#     model_path = "../../codebert-custom3/full_model.pth"
#     tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#     model = torch.load(model_path, map_location="cpu", weights_only=False)
#     return tokenizer, model

@pytest.mark.parametrize("entry", test_data)
def test_codeT5_evaluation_runs(entry, codeT5):
    print("----------------- CodeT5 ---------------------")
    tokenizer, model = codeT5
    evaluate_seq_2_seq(entry, tokenizer, model)

# @pytest.mark.parametrize("entry", test_data)
# def test_codeBERTGPT2_evaluation_runs(entry, codeBERTGPT2):
#     print("------------------ CodeBERT-GPT2 -----------------")
#     tokenizer, model = codeBERTGPT2
#     evaluate_seq_2_seq(entry, tokenizer, model)

# @pytest.mark.parametrize("entry", test_data)
# def test_codeBERTcodeBERT_evaluation_runs(entry, codeBERTcodeBERT):
#     print("------------------ CodeBERT-CodeBERT -----------------")
#     tokenizer, model = codeBERTcodeBERT
#     evaluate_seq_2_seq(entry, tokenizer, model)



# @pytest.mark.parametrize("entry", test_data)
# def test_codeBERTcodeBERT_evaluation_runs(entry, codeBERTcustom0):
#     print("------------------ CodeBERT-Custom0 -----------------")
#     tokenizer, model = codeBERTcustom0
#     evaluate_custom(entry, model, tokenizer)


# @pytest.mark.parametrize("entry", test_data)
# def test_codeBERTcodeBERT_evaluation_runs(entry, codeBERTcustom1):
#     print("------------------ CodeBERT-Custom1 -----------------")
#     tokenizer, model = codeBERTcustom1
#     evaluate_custom(entry, model, tokenizer)

# @pytest.mark.parametrize("entry", test_data)
# def test_codeBERTcodeBERT_evaluation_runs(entry, codeBERTcustom2):
#     print("------------------ CodeBERT-Custom2 -----------------")
#     tokenizer, model = codeBERTcustom2
#     evaluate_custom(entry, model, tokenizer)

# @pytest.mark.parametrize("entry", test_data)
# def test_codeBERTcodeBERT_evaluation_runs(entry, codeBERTcustom3):
#     print("------------------ CodeBERT-Custom3 -----------------")
#     tokenizer, model = codeBERTcustom3
#     evaluate_custom(entry, model, tokenizer)
