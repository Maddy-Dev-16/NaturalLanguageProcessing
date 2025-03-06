import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
import numpy as np

# Load vocabulary
def loadVocabulary(vocabFile):
    with open(vocabFile, "rb") as file:
        wordToIndex, indexToWord = pickle.load(file)
    return wordToIndex, indexToWord

# Define FFNN model
class FeedForwardLM(nn.Module):
    def __init__(self, vocabSize, embedDim, contextSize, hiddenDim):
        super(FeedForwardLM, self).__init__()
        self.embeddings = nn.Embedding(vocabSize, embedDim)
        self.linear1 = nn.Linear(contextSize * embedDim, hiddenDim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hiddenDim, vocabSize)

    def forward(self, inputs):
        embedded = self.embeddings(inputs)
        flat = embedded.view(embedded.size(0), -1)
        hidden = self.activation(self.linear1(flat))
        logits = self.linear2(hidden)
        return logits

# Define RNN model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocabSize, embedDim, hiddenDim):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embedDim)
        self.rnn = nn.RNN(embedDim, hiddenDim, batch_first=True)
        self.fc = nn.Linear(hiddenDim, vocabSize)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        outputs, _ = self.rnn(embedded)
        logits = self.fc(outputs[:, -1, :])  # Get last timestep output
        return logits

# Define LSTM model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocabSize, embedDim, hiddenDim):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embedDim)
        self.lstm = nn.LSTM(embedDim, hiddenDim, batch_first=True)
        self.fc = nn.Linear(hiddenDim, vocabSize)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        outputs, (h_n, c_n) = self.lstm(embedded)
        logits = self.fc(outputs[:, -1, :])  # Get last timestep output
        return logits

# Function to predict the next word
def predictNextWord(model, sentence, wordToIndex, indexToWord, k, lm_type, n_gram_size=None):
    model.eval()
    words = sentence.strip().split()
    
    if lm_type == "-f":
        if n_gram_size is None:
            raise ValueError("n_gram_size is required for FFNN.")
        
        context = words[-(n_gram_size - 1):] if len(words) >= (n_gram_size - 1) else ["<PAD>"] * ((n_gram_size - 1) - len(words)) + words
        input_indices = [wordToIndex.get(w, wordToIndex["<UNK>"]) for w in context]
    else:
        input_indices = [wordToIndex.get(w, wordToIndex["<UNK>"]) for w in words]
    
    input_tensor = torch.tensor([input_indices], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        top_k_probs, top_k_indices = torch.topk(probabilities, k + 1)  # Get extra to filter <UNK>
    
    predictions = [(indexToWord[idx.item()], prob.item()) for idx, prob in zip(top_k_indices, top_k_probs) if indexToWord[idx.item()] != "<UNK>"]
    
    return predictions[:k]  # Return only top k after filtering <UNK>

# Main function
def main():
    print("Usage: python3 generator.py <lm_type> <vocab.pkl> <k> [n_gram_size]")
    print("lm_type: -f for FFNN, -r for RNN, -l for LSTM")
    print("vocab.pkl: Vocabulary file (e.g., vocabPride.pkl or vocabUlysses.pkl)")
    print("k: Number of candidates for next word")
    print("n_gram_size: (Only for FFNN) Use 3 or 5")

    if len(sys.argv) < 4 or (sys.argv[1] == "-f" and len(sys.argv) != 5):
        sys.exit("Invalid arguments")

    lm_type = sys.argv[1]
    vocabFile = sys.argv[2]
    k = int(sys.argv[3])
    n_gram_size = int(sys.argv[4]) if lm_type == "-f" else None

    if lm_type not in ["-f", "-r", "-l"]:
        sys.exit("Invalid model type. Use -f for FFNN, -r for RNN, or -l for LSTM.")

    if not vocabFile.endswith(".pkl"):
        sys.exit("Error: The vocabulary file must be a .pkl file.")

    try:
        wordToIndex, indexToWord = loadVocabulary(vocabFile)
    except FileNotFoundError:
        sys.exit(f"Error: Vocabulary file '{vocabFile}' not found.")

    vocabSize = len(wordToIndex)
    dataset = "Pride&Prejudice" if "Pride" in vocabFile else "Ulysses"

    try:
        if lm_type == "-f":
            if n_gram_size not in [3, 5]:
                sys.exit("Invalid n_gram_size for FFNN. Use 3 or 5.")
            model_path = f"FFNN/{dataset}/{n_gram_size}gram/ffnn_{n_gram_size}.pth"
            model = FeedForwardLM(vocabSize, embedDim=64, contextSize=n_gram_size - 1, hiddenDim=64)
        elif lm_type == "-r":
            model_path = f"RNN/{dataset}/rnn.pth"
            model = RNNLanguageModel(vocabSize, embedDim=100, hiddenDim=128)
        elif lm_type == "-l":
            model_path = f"LSTM/{dataset}/lstm.pth"
            model = LSTMLanguageModel(vocabSize, embedDim=100, hiddenDim=128)
        else:
            sys.exit("Invalid model type.")

        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    except FileNotFoundError:
        sys.exit(f"Error: Model file '{model_path}' not found.")

    print("Model loaded. Enter a sentence to predict the next word:")
    while True:
        sentence = input("Input sentence: ").strip()
        if not sentence:
            print("Error: Input sentence cannot be empty.")
            continue

        predictions = predictNextWord(model, sentence, wordToIndex, indexToWord, k, lm_type, n_gram_size)
        print("Output:")
        for word, prob in predictions:
            print(f"{word} {prob:.4f}")

if __name__ == "__main__":
    main()
