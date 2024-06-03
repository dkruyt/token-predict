import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from termcolor import colored

def load_model_and_tokenizer(model_name, device):
    # Load the pre-trained language model and tokenizer for the given model name
    # and move the model to the specified device (CPU, GPU, or MPS).
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict_next_word(text, model, tokenizer, top_k, device):
    # Tokenize the input text and convert it to tensor format for the model
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Get model predictions without computing gradients
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    # Find the token with the highest probability (predicted next word)
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_token = tokenizer.decode(predicted_index)
    
    # Get the top k tokens with the highest probabilities
    predicted_indices = torch.topk(predictions[0, -1, :], top_k).indices
    predicted_tokens = [tokenizer.decode(idx.item()) for idx in predicted_indices]
    
    # Compute softmax probabilities for the predicted tokens
    predicted_scores = torch.softmax(predictions[0, -1, :], dim=-1)
    predicted_probabilities = [predicted_scores[idx].item() for idx in predicted_indices]
    
    return {
        'input_text': text,
        'input_ids': input_ids.tolist()[0],
        'predicted_next_word': predicted_token,
        'top_predictions': list(zip(predicted_tokens, predicted_probabilities))
    }

def colorize_tokens(input_text, input_ids, tokenizer):
    # Define a color map for visualizing tokens
    color_map = {
        0: 'yellow', 1: 'green', 2: 'red', 3: 'magenta', 4: 'cyan'
    }
    
    # Convert input IDs back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Assign colors to tokens in a cyclic manner based on their positions
    token_colors = [color_map[i % len(color_map)] for i in range(len(tokens))]
    
    colored_text = ""
    colored_ids = ""
    
    for i, token in enumerate(tokens):
        # Handle space token by adding space between tokens
        if token.startswith("Ġ"):
            if i != 0:
                colored_text += " "
                colored_ids += " "
            token = token[1:]  # Remove the space symbol
            
        # Add the token and its ID with colorization
        colored_text += colored(token, token_colors[i])
        colored_ids += colored(str(input_ids[i]), token_colors[i])
    
    return colored_text, colored_ids

def display_predictions(predictions, tokenizer):
    # Colorize the input text and its token IDs
    input_text_colored, input_ids_colored = colorize_tokens(predictions['input_text'], predictions['input_ids'], tokenizer)
    
    # Combine input text with the predicted next word to form a complete sentence
    predicted_sentence = predictions['input_text'] + predictions['predicted_next_word']
    
    # Display the colorized input text and token IDs
    print("\nInput Text: ", input_text_colored)
    print("Token IDs:  ", input_ids_colored)
    
    # Display the top token predictions with their probabilities
    print("\nToken Predictions:")
    for token, score in predictions['top_predictions']:
        print(f"{token} : {score:.2f}")

    # Display the predicted next word with colorization
    print("\nPredicted Next Word:", colored(predictions['predicted_next_word'], 'cyan'))

    # Display the predicted sentence
    print("\nPredicted Sentence: ", predicted_sentence)

if __name__ == "__main__":
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Predict the next word in a sentence using a causal language model.')
    parser.add_argument('text', type=str, help='The input text to predict the next word for.')
    parser.add_argument('--top_k', type=int, default=5, help='The number of top predictions to display (default: 5).')
    parser.add_argument('--model', type=str, default='microsoft/phi-1', help='The pre-trained model to use (default: microsoft/phi-1).')
    
    # Parse the arguments provided by the user
    args = parser.parse_args()
    
    # Check for the availability of CUDA (GPU) or MPS (Apple Metal) for model execution
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Metal Performance Shaders (MPS) is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # Load the model and tokenizer with the specified or default model name
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    # Print the model information
    print("\nModel Information:")
    print(model)
    
    # Predict the next word based on the input text, and display the results
    predictions = predict_next_word(args.text, model, tokenizer, args.top_k, device)
    display_predictions(predictions, tokenizer)