# PyFixAI
An AI-powered tool that detects syntax and logical errors in Python code and suggests intelligent fixes. 
___

## Abstract 
Debugging is one of the most time-consuming and challenging tasks for developers, often requiring significant manual effort to identify and resolve issues. This project proposes an intelligent code debugger, designed to automatically detect both syntax and logical errors in code, while providing suggestions for fixes. The tool will identify common issues such as missing colons and brackets, undefined variables, index errors, and indentation faults. This will enhance productivity by reducing debugging time. Initially, the tool will support Python, with a web-based interface that allows users to seamlessly input their code, quickly identify errors, and receive real-time suggestions for corrections. For this project, we will fine-tune **CodeBERT** using the dataset mentioned below to optimize its performance for our specific task. 
___
 
## Dataset 
The dataset will be sourced from [**Python CodeNet Dataset**](https://github.com/google-research/runtime-error-prediction/).
___

## Benchmarks and Evaluation Metrics 
To assess the performance and effectiveness of the system, a combination of evaluation methods and benchmarks will be utilized as listed below: 
1. Human Evaluation (Manual): We will manually evaluate the output to assess its accuracy, relevance, and overall quality. This qualitative analysis provides valuable insights into the system's performance from a human perspective.  
2. Automated Code Testing (Pytest): Pytest will be employed to conduct automated testing of the generated code. This includes verifying functionality, and correctness ensuring the code meets the required standards.  
3. Pylint Analysis: Pylint, a comprehensive code analyzer, will be used to evaluate code quality by assigning a score based on factors such as readability, maintainability, and adherence to coding standards. We will compare Pylint scores for the code before and after processing through our model, enabling a quantitative analysis of the model's impact on code quality.
___

## Weekly Milestones
| Week Number  | Milestone Details | Milestone Progress | Challenges Faced |
| ------------- | ------------- | ------------- | ------------- |
| Week 10  | Preparing and Preprocessing dataset  | In Progress | The dataset initially proposed could not be used because the proposed fixes in the dataset had errors in them. Therefore, we had to spend additional time in finding and understanding new dataset. |
| Week 11  | Understanding and Finetuning CodeBERT  |  |  |
