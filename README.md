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
- We choose to represent the asnwer candidates (the spans) as a mask matrix of size (MAX_LEN_PAD, 512), where MAX_LEN_PAD is 100 (on average each contexts has around 200 spans selected as possible answers) and 512 is the tokenized sequence length. We chose 100 to minimize the number of the matrix. The mask represents the positions of span in the context, as tokens. 
- Our model computes the product between the DeBERTa model embedding applied to the context and the mask matrix, followed by 3 Dense Layers.
- Ultimately it assigns one of the 4 labels previously mentioned.
- The training, validation, and test datasets are the ones obtained in the Prepare Data section.
- We compared DeBERTa-classification performance with the labels we established as ground truth obtaining a micro-F1 Score
of 0.727.

## Test Data & BLEURT
For testing we prepare the following data: 
- NER extracted candidates, applied to the contexts from the FairytaleQA Dataset.
- Llama-Agen (Llame 3.2 3B fine-tuned for Answer Generation).
- DeBERTa-classification top 10 applied to our test dataset. 

For BLEURT comparison of the above 3 methods with the ground truth answers from FiarytaleQA we generate tuples of shape (label, candidate, BLEURT score) where label is the answer in the original dataset. We constructed a weighted graph to compute the maximum weight matching, ensuring that
each entry is only used once. We compute the average score obtained for each context at the dataset level. 

### ELO Ranking
- We use Llama 3.3 70B as a judge model for the followinng pairs: (DeBERTa-classification top 10, ground truth in Fairytale QA), (DeBERTa-classification top 10, Llama-Agen), (Llama-Agen, ground truth in Fairytale QA).
- We ask the judge model to decide which pair question-answer is better for the provided context, making sure for each match we generate all pairs possible.
- The prompt used can be found in evaluation/prompts.py.
- We pass the pair randomly to avoid bias.
- We use a ELO Ranking system with K=32 and INIT_SCORE=1500.
- We compute the score for individual matches and an overall score.
- Our model (DeBERTa-classification top 10) lost to the Ground Truth pairs, but won againts the fine-tuned Llama.
