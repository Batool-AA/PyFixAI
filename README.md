# PyFixAI
An AI-powered tool that detects syntax and logical errors in Python code and suggests intelligent fixes. 
___

## Abstract 
Debugging is one of the most time-consuming and challenging tasks for developers, often requiring significant manual effort to identify and resolve issues. This project proposes an intelligent code debugger, designed to automatically detect both syntax and logical errors in code, while providing suggestions for fixes. The tool will identify common issues such as missing colons and brackets, undefined variables, index errors, and indentation faults. This will enhance productivity by reducing debugging time. The tool will support Python, with a web-based interface that allows users to seamlessly input their code, quickly identify errors, and receive real-time suggestions for corrections. For this project, we used CodeBERT as our encoder and developed a custom decoder, following the transformer architecture.
___

## Features

- **Transformer-based Code Fixing**  
  Uses CodeBERT as the encoder and a custom decoder (or GPT-2) to correct Python code errors.

- **Content Safety Layer**  
  Detects and filters biased language, hateful content, and sensitive data before processing.

- **Model Benchmarking**  
  Multiple models trained and evaluated on metrics like exact match, BLEU score, AST similarity, and Levenshtein distance.

- **Web Integration**  
  Integrated into a web application to allow users to input buggy code and receive smart fixes in real time.

---

## Methodology

### A. Dataset
- **Source**: [Project CodeNet](https://github.com/IBM/Project_CodeNet)
- **Language Focus**: Python only
- **Data Size**:
  - Training: 180,000 samples
  - Validation: 180,000 samples
  - Testing: 50,000 samples

**Each sample includes:**
- `src`: Buggy code  
- `tgt`: Fixed code  
- `src_verdict`: Error classification  
- `src_id`, `tgt_id`: Identifiers


### B. Models
Two architectures were explored:
- **CodeBERT + Custom Decoder**: A six-layer transformer decoder trained specifically for this task.
- **CodeBERT + GPT-2**: A pre-trained GPT-2 used as a decoder with CodeBERT encoder.


### C. Experimental Setup

- **Platform**: Kaggle Notebooks  
- **GPU**: Dual NVIDIA T4 (16 GB)  
- **Training Framework**: PyTorch + Hugging Face Transformers  
- **Optimizer**: AdamW  
- **Loss Function**: CrossEntropyLoss  
- **Token Limit**: 512 tokens per input sequence

---

## Results

### A. Evaluation Metrics

| Metric                  | Description                                                |
|-------------------------|------------------------------------------------------------|
| **Exact Match**         | Checks character-for-character match of output code        |
| **Compilation Success** | Verifies if the generated code compiles                    |
| **Levenshtein Distance**| Measures how different the output is from ground truth     |
| **AST Similarity**      | Compares code structure and logic via ASTs                 |
| **BLEU Score**          | Measures n-gram overlap between generated and target code  |


### B. Model Performance 

| Model                  | Exact Match | Compilation Success  | Levenshtein Distance  | AST Similarity | BLEU Score |
|------------------------|-------------|----------------------|-----------------------|----------------|------------|
| **CodeT5**             | 0.00        | 0.66                 | 0.89                  | 0.26           | 0.04       |
| **CodeBERT + GPT-2**   | 0.00        | 0.00                 | 9.54                  | 0.00           | 0.02       |
| **CodeBERT + Custom0** | **0.33**    | **1.00**             | **0.14**              | **0.92**       | **0.76**   |
| **CodeBERT + Custom1** | 0.33        | 0.50                 | 0.24                  | 0.49           | 0.61       |
| **CodeBERT + Custom2** | 0.00        | 0.33                 | 0.27                  | 0.26           | 0.45       |
| **CodeBERT + Custom3** | 0.16        | 0.16                 | 4.80                  | 0.16           | 0.50       |

### C. Best Performing Model: `CodeBERT + Custom0`
- Highest scores in **Exact Match**, **Compilation Success**, **AST Similarity**, and **BLEU Score**.
- Lowest **Levenshtein Distance**, indicating fewer changes required to match the correct code.

---

## Integration into the Web Application
The best-performing model (Custom0) was integrated into a web interface where users can input code for debugging. A content safety layer ensures the input is free from harmful or sensitive content before processing.

--- 

## Weekly Milestones
| Week Number  | Milestone Details | Milestone Progress | Challenges Faced |
| ------------- | ------------- | ------------- | ------------- |
| Week 10  | Preparing and Preprocessing Dataset  | Complete | The dataset initially proposed could not be used because the proposed fixes in the dataset had errors in them. Therefore, we had to spend additional time in finding and understanding new dataset. |
| Week 11  | Preparing and Preprocessing Dataset II | Complete | The dataset size was quite large to be used as it is. |
| Week 12 | Understanding and Implementing CodeBERT Model | Complete | None |
| Week 13 | Model Training | Complete | Our devices did not have enough computational power and the online notebooks like Kaggle only provided GPU access for limited time. |
| Week 14 | Model Training + Evaluation | Complete | Our devices did not have enough computational power and the online notebooks like Kaggle only provided GPU access for limited time. |
| Week 15 | Model Evaluation + Website Development | Complete | Open AI API key issues. |
| Week 16 | Project Report and Presentation | Complete | None |

---

## Conclusion
PyFix AI effectively automates code debugging using transformer models. CodeBERT + Custom0 showed the best performance. Future improvements could come from more compute resources and extended training time.

---
