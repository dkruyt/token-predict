from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None
device = None

def initialize_model(model_name='microsoft/phi-1'):
    """Initialize the model and tokenizer on startup"""
    global model, tokenizer, device

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

    # Load the pre-trained language model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully!")

def predict_next_word(text, top_k=5):
    """Predict the next word based on input text"""
    global model, tokenizer, device

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

    # Convert input IDs back to tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

    return {
        'input_text': text,
        'tokens': tokens,
        'input_ids': input_ids.tolist()[0],
        'predicted_next_word': predicted_token,
        'predicted_sentence': text + predicted_token,
        'top_predictions': [
            {'token': token, 'probability': prob}
            for token, prob in zip(predicted_tokens, predicted_probabilities)
        ]
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        top_k = int(data.get('top_k', 5))

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Get predictions
        predictions = predict_next_word(text, top_k)

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })

if __name__ == '__main__':
    # Initialize the model before starting the server
    initialize_model()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5010)
