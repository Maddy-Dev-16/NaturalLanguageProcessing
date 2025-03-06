# **Assignment 2: Neural Language Modeling**  
**Course:** Introduction to NLP   

## **Overview**  
This assignment implements neural language models using **PyTorch** for next-word prediction:  
- **FFNN (Feed Forward Neural Network)**  
- **RNN (Recurrent Neural Network)**  
- **LSTM (Long Short-Term Memory)**  

The models are trained on **Pride and Prejudice** & **Ulysses**, evaluated using **perplexity**.

---

## **Setup & Execution**  
1. Install dependencies:  
   ```bash
   pip install torch numpy
   ```  
2. Run the generator:  
   ```bash
   python3 generator.py <lm_type> <corpus_path> <k>
   ```  
   - `-f` → FFNN  
   - `-r` → RNN  
   - `-l` → LSTM  
   - `<corpus_path>` → Path to dataset  
   - `<k>` → Number of top predictions  

   **Example:**  
   ```bash
   python3 generator.py -f ./corpus.txt 3
   ```

---

## **Files & Submission**  
📂 **generator.py** → Prediction script  
📂 **report.pdf** → Perplexity analysis  
📂 **Pretrained Models** → [Download Here](https://drive.google.com/drive/folders/1ZZEtKmtZDTgTsvdgmL3KbwHtc13i4mra?usp=drive_link)  
📂 **README.md** → Execution guide  

📌 **Zip file submission:** `2024201044_assignment2.zip`  

---

## **Results & Observations**  
✔ **LSTMs perform best** for long sequences  
✔ **RNNs work well but struggle with long-term dependencies**  
✔ **FFNNs overfit easily with large embeddings**  
✔ **Traditional n-grams fail against deep learning models**  

---

## **Author**  
✍️ **Vaibhav Gupta**  
📌 **Roll Number:** 2024201044