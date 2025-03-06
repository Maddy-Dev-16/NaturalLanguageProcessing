import re
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import numpy as np
import sys
import codecs
import random
import matplotlib
matplotlib.use('Qt5Agg')  # or 'wxAgg'
import matplotlib.pyplot as plt
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

def preprocess_pride_and_prejudice(text):
    """Preprocess text specifically for 'Pride and Prejudice'."""
    match = re.search(r'CHAPTER\s*I.', text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    match = re.search(r"Transcriber's\s*Note[:.]?", text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    text = re.sub(
        r'END OF (VOL\.\s*[IVXLCDM]+|THE\s+\w+\s+VOLUME)\s*.*?PRIDE & PREJUDICE\.',
        '', 
        text, 
        flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r'CHAPTER\s*[IVXLCDM\d]+\.\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*+(\s+\*+)+', '', text)

    # Replace - with space and remove _ from the text
    text = text.replace('-', ' ').replace('_', '')
    return text

def preprocess_ulysses(text):
    """Preprocess text specifically for 'Ulysses'."""
    start_index = text.find("[ 1 ]")
    if start_index != -1:
        text = text[start_index:]
    start_index = text.find("[ 1 ]")
    if start_index != -1:
        text = text[start_index:]
    # text = re.sub(r'—\s*i+\s*—', '', text)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    end_index = text.find("Trieste-Zurich-Paris")
    if end_index != -1:
        text = text[:end_index + len("Trieste-Zurich-Paris")]
    return text



def generate_ngram_model(N, tokens):
    """Generate an N-gram model."""
    
    ngram_map = defaultdict(int)
    
    for j in range(len(tokens)):
        if len(tokens[j]) < N:
            tokens[j] = ['<s>'] * (N - len(tokens[j])) + tokens[j]
        for i in range(len(tokens[j]) - N + 1):
            ngram_key = tuple(tokens[j][i:i+N])
            ngram_map[ngram_key] += 1
    return ngram_map

def laplace_smoothing(ngram_model, history_before_ngram, vocab_size):
    """Apply Laplace Smoothing to the N-gram model."""

    # Calculating  the probability of each sentence using Laplace Smoothing


    ngram_model_prob = {}
    for ngram, count in ngram_model.items():
        history = ngram[:-1]
        history_count = history_before_ngram[history]
        ngram_model_prob[ngram] = (count + 1) / (history_count + vocab_size)
    return ngram_model_prob




def linear_regression(x, y):
    """
    Perform simple linear regression to find the slope and intercept.
    x: Independent variable (log counts)
    y: Dependent variable (log frequencies)
    Returns:
        a (slope)
        b (intercept)
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute slope (a)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    a = numerator / denominator

    # Compute intercept (b)
    b = y_mean - a * x_mean
    print(f"Slope: {a}, Intercept: {b}")
    return a, b


def good_turing_smoothing_with_regression(ngram_model, vocab_size, N):
    """Apply Good-Turing smoothing with regression for higher counts."""
    count_of_counts = defaultdict(lambda:0)

    # Step 1: Count the number of N-grams with each frequency
    for count in ngram_model.values():
        count_of_counts[count] += 1
    
    # print the count of counts
    #print(count_of_counts)
    
    

    change_point = 5; # Default value for max_count

    # Plot the histogram of counts and then taking the value of max_count
    plt.bar(list(count_of_counts.keys()), list(count_of_counts.values()))
    plt.ylim(0, 2000)
    plt.xlim(0, 60)
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of N-gram Counts')
    plt.show()

    # Now calculating the count from where the frequency of previous bar < frequency of current bar
    pairs = sorted(zip(count_of_counts.keys(), count_of_counts.values()))
    x,y = zip(*pairs)
    print(sorted(count_of_counts.keys()))
    for i in range(1,len(y)):
        if y[i-1] < y[i]:
            change_point = i
            break
    #print(change_point)
    count_of_counts[0] = (vocab_size ** N) - sum(ngram_model.values())



    # Step 2: Prepare data for regression
    counts = np.array(list(count_of_counts.keys()), dtype=float)
    frequencies = np.array(list(count_of_counts.values()), dtype=float)

    # Exclude counts with 0 frequency for log-log regression
    valid_indices = (counts > 0) & (frequencies > 0)
    log_counts = np.log(counts[valid_indices])
    log_frequencies = np.log(frequencies[valid_indices])

    # Fit a regression model
    a, b = linear_regression(log_counts, log_frequencies)

    # Step 3: Adjust counts using regression for higher frequencies
    adjusted_counts = {}
    total_ngrams = sum(ngram_model.values())

    for c in count_of_counts.keys():
        max_count = max(count_of_counts.keys())
        if c <= change_point:
            # Use original Good-Turing formula for low counts
            next_count = c + 1
            N_c = count_of_counts[c]    # Number of N-grams with count c
            N_next = count_of_counts[next_count] # Number of N-grams with count c+1
            if N_next > 0:
                adjusted_counts[c] = (next_count * N_next) / N_c  # Effective count
                 
            else:
                # If N_{c+1} is 0, find effective count for N_{c+1} using regression
                log_N_next = b + a * np.log(next_count)
                
                N_next_estimated = np.exp(log_N_next)
                
                count_of_counts[next_count] = N_next_estimated
                
                # Now this is the effective count for N_{c+1}
                # update the N_{c+1} count 
                adjusted_counts[c] = next_count * count_of_counts[next_count] / N_c
                


        else:
            # Use regression to estimate N_c and N_{c+1} for high counts
            if(c == max_count):
                log_N_c = b + a * np.log(c)
                N_c_estimated = np.exp(log_N_c)
                adjusted_counts[c] = N_c_estimated
            else:
                log_N_c = b + a * np.log(c)
                log_N_next = b + a * np.log(c + 1)
                N_c_estimated = np.exp(log_N_c)
                
                N_next_estimated = np.exp(log_N_next)
                
                adjusted_counts[c] = (c + 1) * N_next_estimated / N_c_estimated
            


    # Step 4: Calculate adjusted probabilities
    total_adjusted_count = sum(adjusted_counts.get(count, 0) * count_of_counts[count]
                               for count in adjusted_counts)
    
    
    
    # Probabilities = adjusted counts / No of bigrams in the corpus
    adjusted_probabilities = {}
    for c in sorted(adjusted_counts.keys()):
        if c == 0:
            adjusted_probabilities[c] =  count_of_counts[1]/ total_ngrams
        else:
            adjusted_probabilities[c] = adjusted_counts[c] / total_ngrams

    return adjusted_probabilities


def calculate_interpolated_prob(ngram_key, ngram_model, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh, N, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5):
    if N == 1:
        return lambda_1 * ngram_model[ngram_key] / sum(ngram_model.values())
    elif N == 2:
        return (
                lambda_1 * (ngram_model.get(ngram_key, 0) / ngram_model_h.get(ngram_key[:-1])) + 
                lambda_2 * (ngram_model_h.get(ngram_key[-1])) /  sum(ngram_model_h.values())
        )
    elif N == 3:
        return (
                lambda_3 * (ngram_model_hh.get(ngram_key[-1], 0) / sum(ngram_model_hh.values())) + 
                lambda_2 * ngram_model_h.get(ngram_key[-2:], 0) / ngram_model_hh.get(ngram_key[-1:], 1) + 
                lambda_1 * ngram_model[ngram_key] / ngram_model_h.get(ngram_key[:-1], 1)
        )
    elif N == 4:
        return (
                lambda_4 * (ngram_model_hhh.get(ngram_key[-1], 0) / sum(ngram_model_hhh.values())) + 
                lambda_3 * ngram_model_hh.get(ngram_key[-2:], 0) / ngram_model_hhh.get(ngram_key[-1], 1) + 
                lambda_2 * ngram_model_h.get(ngram_key[-3:], 0) / ngram_model_hh.get(ngram_key[-2:], 1) + 
                lambda_1 * ngram_model[ngram_key] / ngram_model_h.get(ngram_key[:-1], 1)
        )
    elif N == 5:
        return (
            lambda_5 * (ngram_model_hhhh.get(ngram_key[-1], 0) / sum(ngram_model_hhhh.values())) + 
                lambda_4 * ngram_model_hhh.get(ngram_key[-2:], 0) / ngram_model_hhhh.get(ngram_key[-1], 1) + 
                lambda_3 * ngram_model_hh.get(ngram_key[-3:], 0) / ngram_model_hhh.get(ngram_key[-2:], 1) + 
                lambda_2 * ngram_model_h.get(ngram_key[-4:], 0) / ngram_model_hh.get(ngram_key[-3:], 1) + 
                lambda_1 * ngram_model[ngram_key] / ngram_model_h.get(ngram_key[:-1], 1)
        )

def interpolation_smoothing(ngram_model, tokens, vocab_size, N):
    ngram_model_prob = {}
    ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh = None, None, None, None
    lambda_1, lambda_2,lambda_3,lambda_4,lambda_5 = 0.2, 0.2, 0.2, 0.2, 0.2
    if(N == 1):
        lambda_1,lambda_2,lambda_3,lambda_4,lambda_5 = 1,0,0,0,0
    elif(N == 2):
        lambda_1,lambda_2,lambda_3,lambda_4,lambda_5 = 0.5,0.5,0,0,0
        ngram_model_h = generate_ngram_model(N-1, tokens)
    elif(N == 3):
        lambda_1,lambda_2,lambda_3,lambda_4,lambda_5 = 0.33,0.33,0.33,0,0
        ngram_model_h = generate_ngram_model(N-1, tokens)
        
        ngram_model_hh = generate_ngram_model(N-2, tokens)
        print(ngram_model_hh)
    elif(N == 4):
        lambda_1,lambda_2,lambda_3,lambda_4,lambda_5 = 0.25,0.25,0.25,0.25,0
        ngram_model_h = generate_ngram_model(N-1, tokens)
        ngram_model_hh = generate_ngram_model(N-2, tokens)
        ngram_model_hhh = generate_ngram_model(N-3, tokens)
    elif(N == 5):
        lambda_1,lambda_2,lambda_3,lambda_4,lambda_5 = 0.2,0.2,0.2,0.2,0.2
        ngram_model_h = generate_ngram_model(N-1, tokens)
        ngram_model_hh = generate_ngram_model(N-2, tokens)
        ngram_model_hhh = generate_ngram_model(N-3, tokens)
        ngram_model_hhhh = generate_ngram_model(N-4, tokens)
    
        
    
    total_tokens = sum(ngram_model.values())
    for token, count in ngram_model.items():
        if N == 1:
            ngram_model_prob[token] = lambda_1 * count / total_tokens
        
        elif N == 2:
            ngram_model_prob[token] = (
                lambda_1 * (ngram_model.get(token, 0) / ngram_model_h.get(token[:-1])) + 
                lambda_2 * (ngram_model_h.get(token[-1])) /  total_tokens
            )
        
        elif N == 3:
            ngram_model_prob[token] = (
                lambda_3 * (ngram_model_hh.get(token[-1], 0) / sum(ngram_model_hh.values())) + 
                lambda_2 * ngram_model_h.get(token[-2:], 0) / ngram_model_hh.get(token[-1:], 1) + 
                lambda_1 * count / ngram_model_h.get(token[:-1], 1)
            )
        
        elif N == 4:
            ngram_model_prob[token] = (
                lambda_4 * (ngram_model_hhh.get(token[-1], 0) / sum(ngram_model_hhh.values())) + 
                lambda_3 * ngram_model_hh.get(token[-2:], 0) / ngram_model_hhh.get(token[-1], 1) + 
                lambda_2 * ngram_model_h.get(token[-3:], 0) / ngram_model_hh.get(token[-2:], 1) + 
                lambda_1 * count / ngram_model_h.get(token[:-1], 1)
            )
        
        elif N == 5:
            ngram_model_prob[token] = (
                lambda_5 * (ngram_model_hhhh.get(token[-1], 0) / sum(ngram_model_hhhh.values())) + 
                lambda_4 * ngram_model_hhh.get(token[-2:], 0) / ngram_model_hhhh.get(token[-1], 1) + 
                lambda_3 * ngram_model_hh.get(token[-3:], 0) / ngram_model_hhh.get(token[-2:], 1) + 
                lambda_2 * ngram_model_h.get(token[-4:], 0) / ngram_model_hh.get(token[-3:], 1) + 
                lambda_1 * count / ngram_model_h.get(token[:-1], 1)
            )
    
    return ngram_model_prob, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh


# Function for training and testing
def train_and_test(corpus_path, N, input_sentence, lm_type, tokens, flag):
    
    
    
    # Determine vocabulary size
    vocabulary = set()
    for sentence_tokens in tokens:
        vocabulary.update(sentence_tokens)
    vocab_size = len(vocabulary)

    # Step 6: Generate N-gram model
    ngram_model = generate_ngram_model(N, tokens)
    history_before_ngram = generate_ngram_model(N-1, tokens)

    # Step 7: Apply smoothing
    ngram_model_prob = {}
    if lm_type == 'l':  # Laplace smoothing
        ngram_model_prob = laplace_smoothing(ngram_model, history_before_ngram, vocab_size)
    elif lm_type == 'g':  # Good-Turing smoothing
        ngram_model_prob = good_turing_smoothing_with_regression(ngram_model, vocab_size, N)
    elif lm_type == 'i':
        ngram_model_prob, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh = interpolation_smoothing(ngram_model, tokens, vocab_size, N)

    
    # Now calculating the probability and perplexity of training sentences
    
    if flag == 1:
        with open('2024201044_LM4_5_train_perplexity.txt', 'w', encoding = "utf-8") as f:    
            if lm_type == 'l':
                avg_perplexity = 0
                for j in range(len(tokens)):
                    probab = 1
                    perplexity = 0
                    count_of_ngrams_in_sentence = len(tokens[j]) - N + 1
                    
                    for i in range(len(tokens[j]) - N + 1):
                        ngram_key = tuple(tokens[j][i:i+N])
                        if ngram_key in ngram_model_prob:
                            probab *= ngram_model_prob[ngram_key]
                            perplexity += np.log(ngram_model_prob[ngram_key])
                        else:
                            history = ngram_key[:-1]
                            history_count = history_before_ngram[history]
                            probab *= 1 / (history_count + vocab_size)
                            perplexity += np.log(1 / (history_count + vocab_size))
                    # Printing the senence
                    f.write(f"Sentence: {' '.join(tokens[j])}" + "\n")
                    
                    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
                    perplexity = np.exp(perplexity)    
                    f.write(f"Perplexity : {str(perplexity)}" + "\n")
                    avg_perplexity += perplexity
                
                # Average perplexity of the training sentences
                #print(len(tokens))
             
                avg_perplexity = avg_perplexity / len(tokens)
                f.write(f"Average Perplexity of the testing sentences: {str(avg_perplexity)}" + "\n")

            elif lm_type == 'g':
                avg_perplexity = 0
                for j in range(len(tokens)):
                    probab = 1
                    perplexity = 0
                    count_of_ngrams_in_sentence = len(tokens[j]) - N + 1
                    for i in range(len(tokens[j]) - N + 1):
                        ngram_key = tuple(tokens[j][i:i+N])
                        if ngram_key in ngram_model:
                            no_of_times_ngram = ngram_model[ngram_key]
                            probab *= ngram_model_prob[no_of_times_ngram]
                            perplexity += np.log(ngram_model_prob[no_of_times_ngram])
                        else:
                            # N0 = How many N-grams
                            perplexity += np.log(ngram_model_prob[0])
                            probab *= ngram_model_prob[0]
                        
                    f.write(f"Sentence: {' '.join(tokens[j])}" + "\n")
                    
                    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
                    perplexity = np.exp(perplexity)    
                    f.write(f"Perplexity : {str(perplexity)}" + "\n")
                    avg_perplexity += perplexity
                
                #print(len(tokens))
                #print(perplexity)
                avg_perplexity = avg_perplexity / len(tokens)
                f.write(f"Average Perplexity of the testing sentences: {str(avg_perplexity)}" + "\n")

            elif lm_type == 'i':
                avg_perplexity = 0
                for j in range(len(tokens)):
                    probab = 1
                    perplexity = 0
                    count_of_ngrams_in_sentence = len(tokens[j]) - N + 1
                    for i in range(len(tokens[j]) - N + 1):
                        ngram_key = tuple(tokens[j][i:i+N])
                        probab = calculate_interpolated_prob(ngram_key, ngram_model, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh, N, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5)
                        if probab == 0:
                            probab = 1/(vocab_size)
                        count_of_ngrams_in_sentence += 1
                        perplexity += np.log(probab)

                            
                    
                    f.write(f"Sentence: {' '.join(tokens[j])}" + "\n")
                    
                    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
                    perplexity = np.exp(perplexity)    
                    f.write(f"Perplexity : {str(perplexity)}" + "\n")
                    avg_perplexity += perplexity
                
                avg_perplexity = avg_perplexity / len(tokens)
                f.write(f"Average Perplexity of the testing sentences: {str(avg_perplexity)}" + "\n")
    else:
        with open('2024201044_LM4_5_test_perplexity.txt', 'w', encoding="utf-8") as f:    
            if lm_type == 'l':
                avg_perplexity = 0
                for j in range(len(tokens)):
                    probab = 1
                    perplexity = 0
                    count_of_ngrams_in_sentence = len(tokens[j]) - N + 1
                    
                    for i in range(len(tokens[j]) - N + 1):
                        ngram_key = tuple(tokens[j][i:i+N])
                        if ngram_key in ngram_model_prob:
                            probab *= ngram_model_prob[ngram_key]
                            perplexity += np.log(ngram_model_prob[ngram_key])
                        else:
                            history = ngram_key[:-1]
                            history_count = history_before_ngram[history]
                            probab *= 1 / (history_count + vocab_size)
                            perplexity += np.log(1 / (history_count + vocab_size))
                    # Printing the senence
                    f.write(f"Sentence: {' '.join(tokens[j])}" + "\n")
                    
                    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
                    perplexity = np.exp(perplexity)    
                    f.write(f"Perplexity : {str(perplexity)}" + "\n")
                    avg_perplexity += perplexity
                
                # Average perplexity of the training sentences
                #print(len(tokens))
                
                avg_perplexity = avg_perplexity / len(tokens)
                f.write(f"Average Perplexity of the testing sentences: {str(avg_perplexity)}" + "\n")

            elif lm_type == 'g':
                avg_perplexity = 0
                for j in range(len(tokens)):
                    probab = 1
                    perplexity = 0
                    count_of_ngrams_in_sentence = len(tokens[j]) - N + 1
                    for i in range(len(tokens[j]) - N + 1):
                        ngram_key = tuple(tokens[j][i:i+N])
                        if ngram_key in ngram_model:
                            no_of_times_ngram = ngram_model[ngram_key]
                            probab *= ngram_model_prob[no_of_times_ngram]
                            perplexity += np.log(ngram_model_prob[no_of_times_ngram])
                        else:
                            # N0 = How many N-grams
                            perplexity += np.log(ngram_model_prob[0])
                            probab *= ngram_model_prob[0]
                        
                    f.write(f"Sentence: {' '.join(tokens[j])}" + "\n")
                    
                    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
                    perplexity = np.exp(perplexity)    
                    f.write(f"Perplexity : {str(perplexity)}" + "\n")
                    avg_perplexity += perplexity
                
                #print(len(tokens))
                #print(perplexity)
                avg_perplexity = avg_perplexity / len(tokens)
                f.write(f"Average Perplexity of the testing sentences: {str(avg_perplexity)}" + "\n")

            elif lm_type == 'i':
                avg_perplexity = 0
                for j in range(len(tokens)):
                    probab = 1
                    perplexity = 0
                    count_of_ngrams_in_sentence = len(tokens[j]) - N + 1
                    for i in range(len(tokens[j]) - N + 1):
                        ngram_key = tuple(tokens[j][i:i+N])
                        probab = calculate_interpolated_prob(ngram_key, ngram_model, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh, N, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5)
                        if probab == 0:
                            probab = 1/(vocab_size)
                        count_of_ngrams_in_sentence += 1
                        perplexity += np.log(probab)

                            
                    
                    f.write(f"Sentence: {' '.join(tokens[j])}" + "\n")
                    
                    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
                    perplexity = np.exp(perplexity)    
                    f.write(f"Perplexity : {str(perplexity)}"+ "\n")
                    avg_perplexity += perplexity
                
                avg_perplexity = avg_perplexity / len(tokens)
                f.write(f"Average Perplexity of the testing sentences: {str(avg_perplexity)}" + "\n")


    
    # Step 8: Calculate probability of the input sentence
    input_sentence_tokens = word_tokenize(input_sentence.lower())

    # Add start padding for N-grams
    if len(input_sentence_tokens) < N:
        input_sentence_tokens = ['<s>'] * (N - len(input_sentence_tokens)) + input_sentence_tokens

    if lm_type == 'l':
        probab = 1
        perplexity = 0
        count_of_ngrams_in_sentence = 0 
        for i in range(len(input_sentence_tokens) - N + 1):
            ngram_key = tuple(input_sentence_tokens[i:i+N])
           
            if ngram_key in ngram_model_prob:
                probab *= ngram_model_prob[ngram_key]
                perplexity += np.log(ngram_model_prob[ngram_key])
            else:
                history = ngram_key[:-1]
                history_count = history_before_ngram[history]
                probab *= 1 / (history_count + vocab_size)
                perplexity += np.log(1 / (history_count + vocab_size))
            #print(f"ngram_key: {ngram_key}, probab: {probab}, perplexity: {perplexity}")
            count_of_ngrams_in_sentence += 1
    elif lm_type == 'g':
        probab = 1
        perplexity = 0
        count_of_ngrams_in_sentence = 0
        for i in range(len(input_sentence_tokens) - N + 1):
            ngram_key = tuple(input_sentence_tokens[i:i+N])
            if ngram_key in ngram_model:
                no_of_times_ngram = ngram_model[ngram_key]
                perplexity += np.log(ngram_model_prob[no_of_times_ngram])
                probab *= ngram_model_prob[no_of_times_ngram]
            else:
                # N0 = How many N-grams possible - How many N-grams observed in corpus
                N_not = vocab_size ** N - sum(ngram_model.values())
                perplexity += np.log(ngram_model_prob[0])
                probab *= ngram_model_prob[0]
            count_of_ngrams_in_sentence += 1
    elif lm_type == 'i':
        probab = 1
        perplexity = 0
        count_of_ngrams_in_sentence = 0
        for i in range(len(input_sentence_tokens) - N + 1):
            ngram_key = tuple(input_sentence_tokens[i:i+N])
            probab = calculate_interpolated_prob(ngram_key, ngram_model, ngram_model_h, ngram_model_hh, ngram_model_hhh, ngram_model_hhhh, N, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5)
            if probab == 0:
                probab = 1/(vocab_size)
            count_of_ngrams_in_sentence += 1
            perplexity += np.log(probab)
    #print(f"\ncount of ngrams in the sentence: {count_of_ngrams_in_sentence}")
    # Now calculate the perplexitty of the input sentence
    # perplexity = perplexity / 
    #print("count of ngrams in the sentence: ", count_of_ngrams_in_sentence)
    #print("Perplexity before before the input sentence: ", perplexity)
    perplexity = (-1)*perplexity/count_of_ngrams_in_sentence
    #print(f"Perplexity before the input sentence: {perplexity}")
    perplexity = np.exp(perplexity)
    return perplexity
    #print(f"\nProbability of the input sentence: {probab}")



def main(corpus_path, N, input_sentence, lm_type):
    # Step 1: Read the file
    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Step 2: Extract the Title
    title_match = re.search(r'Title:\s*(.+)', text, re.IGNORECASE)
    if not title_match:
        raise ValueError("Title not found in the text.")
    title = title_match.group(1).strip()
    print(f"Processing book: {title}")

    # Step 3: Preprocess the text based on the title
    if title.lower() == "pride and prejudice":
        text = preprocess_pride_and_prejudice(text)
    elif title.lower() == "ulysses":
        text = preprocess_ulysses(text)
    else:
        raise ValueError(f"Unsupported book title: {title}")

    # Convert the text to lowercase
    text = text.lower()

    # Step 4: Sentence Tokenization
    sentences = sent_tokenize(text)

    # Divide in train and test set
    # selecting randomly 1000 sentences for testing

    test_sentences = random.sample(sentences, 1000)
    train_sentences = [sentence for sentence in sentences if sentence not in test_sentences]

     

    # Step 5: Tokenize the text
    all_tokens = [word_tokenize(sentence) for sentence in sentences]

    train_tokens = [word_tokenize(sentence) for sentence in train_sentences]
    test_tokens = [word_tokenize(sentence) for sentence in test_sentences]

    perplexity = train_and_test(corpus_path, N, input_sentence, lm_type, train_tokens, 1)
    perplexity = train_and_test(corpus_path, N, input_sentence, lm_type, test_tokens, 2)    

    print(f"Perplexity of the input sentence: {perplexity}")     
    

if __name__ == "__main__":

    # taking arguments from command line
    if len(sys.argv) == 2:
        lm_type = sys.argv[1]
    else:
        print("Usage: python Q2.py <lm_type>")
        sys.exit(1)
    
    # Path to the corpus file
    corpus_path = input("Enter the path to the corpus file: ")
    
    # N-gram model order
    N = int(input("Enter the value of N for the N-gram model: "))
    
    if lm_type == 'l':
        # Laplace Smoothing
        print("Laplace Smoothing:")
    elif lm_type == 'g':
        # Good Turing Smoothing
        print("Good Turing Smoothing:")
    elif lm_type == 'i':
        # Interpolation Smoothing
        print("Interpolation Smoothing:")
    else:
        print("Invalid smoothing type. Please enter 'l', 'g', or 'i'.")
        sys.exit(1)

    

    
    input_sentence = input("Enter a sentence to calculate its probability: ")
    main(corpus_path, N, input_sentence, lm_type)
