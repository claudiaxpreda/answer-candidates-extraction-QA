# Automated Extraction of Answer Candidates for Question Generation

This repository hosts the code for the automated extraction of answer candidates for question generation. This project is based on the FairytaleQA Dataset. It propose a solution which syntactically parse (constituency tree) the context and uses a DeBERTa-based classification model to assign labels to each span of text obtained. Thus, an extracted candidate can be on of the following: 
- Very Good (Label 1)
- Good (Label 2)
- Average (Label 3)
- Unusable (Label 4)

The repositiry also contains the steps of creating the data for training the classification model, as well as testing its performance. 

## Prepare Data
- Fine-tuning 3 Llama 3.2 3B models for Question Generation, Question Answering, Answer Generation (finetune step) using FairytaleQA Dataset.
- Constituency Parsing for the (train, validation, test) slices using Berkeley Neural Parser (https://spacy.io/universe/project/self-attentive-parser)
- Computing the tokenized span position (start, end) in the tokenized context for passing all candidates in one step.
- Generate the questions for the (span, context) using the QGen fine-tuned model.
- Compute the loss of (generated question, context) with the target being the span(candidate) using the QA fine-tuned model.
- Assign labels for the computed losses.

Original FairytaleQA Dataset -> Parsed Dataset -> Parsed Dataset + Tokens Start/End -> Dataset for the task

## Train DeBERTa Classifier 

## Test Data

## Evaluation 

### BLEURT
### ELO Ranking
