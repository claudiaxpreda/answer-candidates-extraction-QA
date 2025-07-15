JUDGE_PROMPT = """

You will be given a context and two pairs of an question and its answer.
Your task is to establish which one of the pairs is better for the given context.

To determine which pair is better for the given context, please follow these steps:

    1. Carefully read and understand the context provided and the two pairs.
    2. For each pair, identify if the answer given as input answers the question given as input. 
    3. If the question and answer are based on information from the context.
    4. If the question assesses understanding rather than simple recall.
    5. If the pair is more complex rather than straight-forward.



After your analysis, you need to assign a verdict for the first pair, using the following guidance:
    1. Win, if the fist pair is determined to be better. 
    2. Lose, if the second pair is determined to better. 
    3. Tie, if both pairs are deemed to be equal. 
    4. You must declare a tie only when the decision is difficult. 


The format in which the feedback should be provided is:


Feedback:::

Verdict: (if the firts pair is Win, Lose, Tie)

Evaluation: (short explanation of your rationale, as a text)


The input is: 

Context: {context}
Pair1: ({question1}, {answer1})
Pair2: ({question2}, {answer2})

You MUST provide values for 'Evaluation:' and 'Verdict:' in your answer.
Please ensure your analysis is thorough, impartial, and based on the content provided.


"""
