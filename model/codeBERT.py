from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Error or No Error

def classify_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    error_label = torch.argmax(probabilities).item()
    
    return "Error detected" if error_label == 1 else "No error"

code_sample = """n = int ( input ()) 
              l = [ input () for i in range ( n )] 
              for i in range ( 3 ):
                  m = max ( l ) 
                  a . remove ( m ) 
  	              print ( m )"""  # Missing parenthesis
print(classify_code(code_sample))

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# model_name = "Salesforce/codet5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# def generate_fix(buggy_code):
#     inputs = tokenizer(buggy_code, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(**inputs, max_length=512)
#     fixed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return fixed_code

# buggy_code = """n = int ( input ()) 
#              l = [ input () for i in range ( n )] 
#              for i in range ( 3 ):
#                  m = max ( l ) 
#                  a . remove ( m ) 
# 	             print ( m )"""

# fixed_code = generate_fix(buggy_code)
# print("Suggested Fix:", fixed_code)
