import nltk
from tokenizer import tokenize
from nltk.tokenize import word_tokenize
from language_model import laplace_smoothing,good_turing_smoothing_with_regression,generate_ngram_model, calculate_interpolated_prob,interpolation_smoothing
from collections import Counter
import sys
import math
import numpy as np

def predict_k_words_laplace(test_sentence,corpus_path,n,k,tokens,all_tokens):
    word_prob = set()
    ngram_model = generate_ngram_model(n,tokens)
    history_before_ngram_model = generate_ngram_model(n-1,tokens)
    vocab = set(ngram_model.keys())
    
    vocab_size = len(vocab)
    test_sentence = test_sentence.lower()
    history_before_ngram = generate_ngram_model(n-1,tokens)
    ngram_model_prob = laplace_smoothing(ngram_model, history_before_ngram_model,vocab_size)
    for word in all_tokens:
        test_string_temp = test_sentence+" "+word
        sentence_tokens = word_tokenize(test_string_temp)
        if len(sentence_tokens) < n:
            sentence_tokens = ['<s>']*(n-len(sentence_tokens)) + sentence_tokens
        
        probab = 1
        count_of_ngrams_in_sentence = 0
        for i in range(len(sentence_tokens) - n + 1):
            ngram_key = tuple(sentence_tokens[i:i+n])
           
            if ngram_key in ngram_model_prob:
                probab *= ngram_model_prob[ngram_key]
            else:
                history = ngram_key[:-1]
                history_count = history_before_ngram[history]
                probab *= 1 / (history_count + vocab_size)
            #print(f"ngram_key: {ngram_key}, probab: {probab}, perplexity: {perplexity}")
            count_of_ngrams_in_sentence += 1
        word_prob.add((probab,word))
    sorted_word_probability = sorted(word_prob, key=lambda x: x[0], reverse=True)
    k=int(k)
    i=int(0)
    for word in sorted_word_probability:
        i+=1
        if i>k:
            break
        print(word[1],word[0])


    

def predict_k_words_good_turing(test_sentence,corpus_path,n,k,tokens,all_tokens):
    word_prob = set()
    ngram_model = generate_ngram_model(n,tokens)
    history_before_ngram_model = generate_ngram_model(n-1,tokens)
    vocab = set(ngram_model.keys())
    
    vocab_size = len(vocab)
    test_sentence = test_sentence.lower()
    history_before_ngram = generate_ngram_model(n-1,tokens)
    ngram_model_prob = good_turing_smoothing_with_regression(ngram_model,vocab_size,n)

    for word in all_tokens:
        test_string_temp = test_sentence+" "+word
        sentence_tokens = word_tokenize(test_string_temp)
        if len(sentence_tokens) < n:
            sentence_tokens = ['<s>']*(n-len(sentence_tokens)) + sentence_tokens
        
        probab = 1
        
        count_of_ngrams_in_sentence = 0
        for i in range(len(sentence_tokens) - n + 1):
            ngram_key = tuple(sentence_tokens[i:i+n])
            if ngram_key in ngram_model:
                no_of_times_ngram = ngram_model[ngram_key]
                
                probab *= ngram_model_prob[no_of_times_ngram]
            else:
                # N0 = How many N-grams possible - How many N-grams observed in corpus
                N_not = vocab_size ** n - sum(ngram_model.values())
                
                probab *= ngram_model_prob[0]
            count_of_ngrams_in_sentence += 1
        word_prob.add((probab,word))
    sorted_word_probability = sorted(word_prob, key=lambda x: x[0], reverse=True)
    k=int(k)
    i=int(0)
    for word in sorted_word_probability:
        i+=1
        if i>k:
            break
        print(word[1],word[0])

def predict_k_words_interpole(test_sentence,corpus_path,n,k,tokens,all_tokens):
    word_prob = set()
    ngram_model = generate_ngram_model(n,tokens)
    vocab = set(ngram_model.keys())
    
    vocab_size = len(vocab)
    test_sentence = test_sentence.lower()
    history_before_ngram = generate_ngram_model(n-1,tokens)
    ngram_model_prob, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh = interpolation_smoothing(ngram_model, tokens, vocab_size, n)
    
    for word in all_tokens:
        test_string_temp = test_sentence+" "+word
        sentence_tokens = word_tokenize(test_string_temp)
        if len(sentence_tokens) < n:
            sentence_tokens = ['<s>']*(n-len(sentence_tokens)) + sentence_tokens
        
        probab = 1
        perplexity = 0
        count_of_ngrams_in_sentence = 0
        for i in range(len(sentence_tokens) - n + 1):
            ngram_key = tuple(sentence_tokens[i:i+n])
            probab = calculate_interpolated_prob(ngram_key, ngram_model, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh, n, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5)
            if probab == 0:
                probab = 1/(vocab_size)
            count_of_ngrams_in_sentence += 1
            perplexity += np.log(probab)
        word_prob.add((probab,word))
    sorted_word_probability = sorted(word_prob, key=lambda x: x[0], reverse=True)
    k=int(k)
    i=int(0)
    for word in sorted_word_probability:
        i+=1
        if i>k:
            break
        print(word[1],word[0])



def main():
    if len(sys.argv) == 2:
        lm_type = sys.argv[1]
    else:
        print("Usage: python Q2.py <lm_type>")
        sys.exit(1)
    
    # Path to the corpus file
    corpus_path = input("Enter the path to the corpus file: ")
    
    # N-gram model order
    n = int(input("Enter the value of n: "))
    k = input("Enter the value of k: ")
    test_sentence=input("Enter a Sentence: ")
    # print(test_sentence)

    # read the corpus
    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # lowercase the corpus
    text = text.lower()

    # tokenize the corpus
    tokens = tokenize(text)

    all_tokens = set(word_tokenize(text))    

    if lm_type=='l':
        predict_k_words_laplace(test_sentence,corpus_path,n,k,tokens,all_tokens)

    elif lm_type=='g':
        predict_k_words_good_turing(test_sentence,corpus_path,n,k,tokens,all_tokens)
    
    elif lm_type=='i':
        predict_k_words_interpole(test_sentence,corpus_path,n,k,tokens,all_tokens)


if __name__ == "__main__":
    # nltk.download('stopwords')
    # nltk.download('punkt')
    main()