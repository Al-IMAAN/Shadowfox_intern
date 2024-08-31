import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def generate_embedding(text: str) -> torch.Tensor:
    """
    Generate BERT embedding for the given text.

    Parameters:
        text (str): The text to generate embedding for.

    Returns:
        torch.Tensor: The BERT embedding for the text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Use the mean of the embeddings as the sentence embedding


def check_consistency_with_bert(prompt: str, response: str) -> float:
    """
    Check the consistency of the response with the prompt by comparing BERT embeddings.

    Parameters:
        prompt (str): The original prompt.
        response (str): The response text.

    Returns:
        float: Cosine similarity score between the prompt and response embeddings.
    """
    prompt_embedding = generate_embedding(prompt)
    response_embedding = generate_embedding(response)
    similarity = cosine_similarity(prompt_embedding.detach().numpy(), response_embedding.detach().numpy())
    return similarity[0][0]


def analyze_prompts_with_bert(prompts: List[str], responses: List[str]) -> List[float]:
    """
    Analyze prompts and responses using BERT embeddings for consistency.

    Parameters:
        prompts (list of str): A list of prompts.
        responses (list of str): A list of corresponding responses.

    Returns:
        list of float: Cosine similarity scores for each prompt-response pair.
    """
    consistency_scores = [check_consistency_with_bert(prompt, response) for prompt, response in zip(prompts, responses)]
    return consistency_scores


def visualize_scores_with_bert(prompts: List[str], consistency_scores: List[float]):
    """
    Visualize consistency scores using bar plots.

    Parameters:
        prompts (list of str): A list of prompts.
        consistency_scores (list of float): Consistency scores for each prompt.
    """
    # Create a DataFrame for visualization
    data = {
        'Prompt': prompts,
        'Consistency Score': consistency_scores
    }
    df = pd.DataFrame(data)

    # Visualization of consistency scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Prompt', y='Consistency Score', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Consistency')
    plt.tight_layout()  # Adjust layout to fit labels and titles
    plt.show()


# Example prompts and corresponding responses (assuming you have responses)
prompts = [
    "Explain the significance of BERT in natural language processing.",
    "What are the benefits of using transformers in NLP?",
    "Compare GPT-3 with other language models like BERT."
]

responses = [
    "BERT has revolutionized NLP by allowing for better context understanding.",
    "Transformers enable parallel processing, which speeds up training.",
    "GPT-3 is larger and more powerful than BERT, allowing for more complex language tasks."
]

# Analyze prompts using BERT
consistency_scores = analyze_prompts_with_bert(prompts, responses)

# Visualize scores
visualize_scores_with_bert(prompts, consistency_scores)

# Print summary of findings
print("Summary of Findings:")
print(f"\nPrompts analyzed: {len(prompts)}")

for idx, prompt in enumerate(prompts):
    print(f"\nPrompt {idx + 1}:")
    print(f" - Prompt: {prompt}")
    print(f" - Response: {responses[idx]}")
    print(f" - Consistency Score: {consistency_scores[idx]}")

print("\nConclusions:")
print("1. Analyze which prompts received higher consistency scores.")
print("2. Discuss any patterns observed in responses based on the type of prompts.")
print("3. Explore potential applications of BERT based on the analysis.")
print("4. Suggest areas for improvement or further research.")
