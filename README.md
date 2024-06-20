# Next Token / Word Predictor

This project predicts the next word in a sentence using a pre-trained causal language model from the Hugging Face Transformers library. The script allows users to input a sentence and get predictions for the next word along with the probabilities of the top predictions.

## Features

- Load pre-trained language models and tokenizers.
- Predict the next word in a given sentence.
- Display the top K token predictions with their probabilities.
- Visualize tokenized input text and token IDs with colorization for better readability.

## Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- Termcolor

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/dkruyt/predict-token.git
    cd next-word-predictor
    ```

2. Install the required packages:
    ```sh
    pip install torch transformers termcolor
    ```

## Usage

To predict the next word in a given sentence, run the script with the desired input text. You can specify the pre-trained model and the number of top predictions to display.

### Example

```sh
python predict-token.py "The quick brown fox" --top_k 5 --model microsoft/phi-1
```

### Arguments

- `text`: The input text to predict the next word for.
- `--top_k`: (Optional) The number of top predictions to display (default: 5).
- `--model`: (Optional) The pre-trained model to use (default: microsoft/phi-1).

## Code Overview

### Functions

- `load_model_and_tokenizer(model_name, device)`: Loads the pre-trained language model and tokenizer.
- `predict_next_word(text, model, tokenizer, top_k, device)`: Predicts the next word and returns the top K predictions.
- `colorize_tokens(input_text, input_ids, tokenizer)`: Colorizes the tokens for better readability.
- `display_predictions(predictions, tokenizer)`: Displays the input text, token IDs, top predictions, and the predicted next word.

### Main Script

The main script sets up argument parsing for command-line arguments, checks for the availability of CUDA or MPS for model execution, loads the model and tokenizer, and displays the model information and predictions.

## Example Output

```plaintext
CUDA is available. Using GPU.

Model Information:
<model details>

Input Text:  The quick brown fox
Token IDs:   1332  6140  1060  3243

Token Predictions:
jumps : 0.45
leaps : 0.10
runs  : 0.05
flies : 0.04
...

Predicted Next Word: jumps

Predicted Sentence:  The quick brown fox jumps
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Termcolor](https://pypi.org/project/termcolor/)
