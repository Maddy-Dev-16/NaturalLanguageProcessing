import nltk
from tokenizer import tokenize, remove_stopwords
from language_model import laplace_smoothing,good_turing_smoothing_with_regression,generate_ngram_model, calculate_interpolated_prob,interpolation_smoothing,generalized_model,frequency_of_frequency,model_train_lambda
from collections import Counter
import sys
import math
import numpy as np

def laplace_model_smoothing_predict(ngrams_counts,ngrams_n_1_counts,total_char,vocab_size,corpus_path,n,test_sentence):
    probability = 1.0
    new_file_name = corpus_path.replace('corpus/', '')
    probability = 1.0   
    perplexity=0.0 
    if n==1:
        test_sentence = test_sentence.lower()
        test_tokens = test_sentence.split()
        for test_token in test_tokens:
            if test_token not in dict(ngrams_counts):
                token_count = 0
            else:
                token_count = dict(ngrams_counts)[test_token]
            probability *= (token_count+1) / (total_char+vocab_size)
            perplexity+= np.log((token_count+1) / (total_char+vocab_size))
        perplexity = (-1)*perplexity/len(test_tokens)

    else:
        test_sentence = test_sentence.lower()
        test_tokens = test_sentence.split()
        if len(test_tokens)<n:
            test_tokens=['<SOS>']*(n-len(test_tokens))+test_tokens
        test_ngrams = list(zip(*[test_tokens[i:] for i in range(n)]))

        for ngram in test_ngrams:
            ngram_count = ngrams_counts[ngram]
            n_1_gram_count = ngrams_n_1_counts[ngram[:-1]]
            probability*= (ngram_count + 1) / (n_1_gram_count + vocab_size)
            perplexity+= np.log((ngram_count + 1) / (n_1_gram_count + vocab_size))
            # print(ngram,"= ", ngram_count,"   ",ngram[:-1],"= ",n_1_gram_count,"Vocab_size= ",vocab_size,"val of N= ",len(test_ngrams)," probability= ",probability," perplexity= ",perplexity)
        # print("length of test_ngram",len(test_ngrams),"   ",perplexity)
        perplexity = (-1)*perplexity/len(test_ngrams)
    perplexity=np.exp(perplexity)
    return probability,perplexity

def find_all_tokens(corpus_path):
    new_file_name = corpus_path.replace('corpus/', '')
    tokens_train_dataset=tokenize(corpus_path, new_file_name)
    all_tokens=set()
    for i in range(len(tokens_train_dataset)):
        for j in range(len(tokens_train_dataset[i])):
            all_tokens.add(tokens_train_dataset[i][j])
    # print(all_tokens)
    return all_tokens

def predict_k_words_laplace(test_sentence,corpus_path,n,k):
    all_tokens=find_all_tokens(corpus_path)
    word_probability=set()
    new_file_name = corpus_path.replace('corpus/', '')
    if n==1:
        ngrams_counts,total_char,vocab_size= unigram_model(corpus_path, new_file_name)
    else:
        ngrams_counts,ngrams_n_1_counts,vocab_size=generalized_model(corpus_path, new_file_name,n)
    for word in all_tokens:
        test_string_temp=test_sentence+" "+word
        if n==1:
            probability_temp=laplace_model_smoothing_predict(ngrams_counts,1,total_char,vocab_size,corpus_path,n,test_string_temp)
        else:
            probability_temp=laplace_model_smoothing_predict(ngrams_counts,ngrams_n_1_counts,1,vocab_size,corpus_path,n,test_string_temp)
        word_probability.add((probability_temp,word))
    sorted_word_probability = sorted(word_probability, key=lambda x: x[0], reverse=True)
    k=int(k)
    i=int(0)
    for word in sorted_word_probability:
        i+=1
        if i>k:
            break
        print(word[1],word[0])

def good_turing_predict(trained_fof_dict,a,b,ngrams_counts,corpus_path,n,test_sentence,total_count):
    probability=1.0
    perplexity=0.0
    # print(type(ngrams_counts))
    test_sentence = test_sentence.lower()
    test_tokens = test_sentence.split()
    if len(test_tokens)<n:
        test_tokens=['<SOS>']*(n-len(test_tokens))+test_tokens
    test_ngrams = list(zip(*[test_tokens[i:] for i in range(n)]))
    for ngram in test_ngrams:
        if n==1:
            ngram_1=ngram[0]
            if ngram_1 not in dict(ngrams_counts):
                r=0
                nominator=trained_fof_dict[1]
                # print("ngrams= ",ngram,"   r=",r,"   Nr= ",trained_fof_dict[1])
            else:
                r= dict(ngrams_counts)[ngram_1]
                # print(r)
                if r+1 not in trained_fof_dict:
                    nr1=math.exp(a + b * np.log(r+1))
                    trained_fof_dict[r+1]=nr1
                else:
                    nr1=trained_fof_dict[r+1]
                nominator= (r+1)*nr1/trained_fof_dict[r]
            probability*=nominator/total_count
            perplexity+= math.log(nominator/total_count)
            # print("ngrams= ",ngram,"   r=",r,"   Nr= ",trained_fof_dict[r],"   Nr+1= ",nr1,"   nominator= ",nominator, "   probability= ",probability,"   perplexity= ",perplexity)
            # print("total word count= ",total_count) 
        else:
            r=ngrams_counts[ngram]
            nominator=0
            if r==0:
                nominator=trained_fof_dict[1]
                # print("ngrams= ",ngram,"   r=",r,"   Nr= ",trained_fof_dict[1])
            else:
                n_r_1=int(r)+1
                # print("n_r_1= ",n_r_1)
                if n_r_1 not in trained_fof_dict:
                    nr1=math.exp(a + b * np.log(r+1))
                    trained_fof_dict[r+1]=nr1
                else:
                    nr1=trained_fof_dict[r+1]
                nominator= (r+1)*trained_fof_dict[r+1]/trained_fof_dict[r]
            probability*=nominator/len(ngrams_counts)

            # perplexity+= math.log(nominator/len(ngrams_counts))
        if probability!=0:
            perplexity=math.log(probability)
            perplexity = (-1)*perplexity/len(test_ngrams)
    perplexity=math.exp(perplexity)
    return probability,perplexity

def predict_k_words_good_turing(test_sentence,corpus_path,n,k):
    all_tokens=find_all_tokens(corpus_path)
    word_probability=set()
    new_file_name = corpus_path.replace('corpus/', '')

    if n==1:
        ngrams_counts,total_char,vocab_size= unigram_model(corpus_path, new_file_name)
    else:
        ngrams_counts,ngrams_n_1_counts,vocab_size=generalized_model(corpus_path, new_file_name,n)


    # _= frequency_of_frequency(ngrams_counts,n)
    trained_fof,a,b= frequency_of_frequency(ngrams_counts,n)
    perplexity=0.0
    trained_fof_dict ={}
    for key,value in trained_fof:
        trained_fof_dict[key]=value
    total_count=0
    if n==1:
        for key in ngrams_counts:
            total_count+= key[1]
            # print("total count= ",total_count)
    else:
        for key in ngrams_counts:
            total_count+= ngrams_counts[key]

    for word in all_tokens:
        test_string_temp=test_sentence+" "+word
        if n==1:
            probability_temp=good_turing_predict(trained_fof_dict,a,b, ngrams_counts,corpus_path,n,test_string_temp,total_count)
        else:
            probability_temp=good_turing_predict(trained_fof_dict,a,b, ngrams_counts,corpus_path,n,test_string_temp,total_count)
        word_probability.add((probability_temp,word))
    sorted_word_probability = sorted(word_probability, key=lambda x: x[0], reverse=True)
    k=int(k)
    i=int(0)
    for word in sorted_word_probability:
        i+=1
        if i>k:
            break
        print(word[1],word[0])

def interpol_model_smoothing_predict(
                ngrams_counts,ngrams_counts1, ngrams_counts2, ngrams_counts3, ngrams_counts4, ngrams_counts5,
                ngrams_n_1_counts2, ngrams_n_1_counts3, ngrams_n_1_counts4, ngrams_n_1_counts5,
                total_char, vocab_size, corpus_path, n, test_string_temp,lambdas
            ):
    probability = 1.0
    perplexity=0.0
    if n==1:
        test_sentence = test_string_temp.lower()
        test_tokens = test_sentence.split()
        test_ngrams=test_tokens
        for test_token in test_tokens:
            if test_token not in dict(ngrams_counts):
                token_count = 0
            else:
                token_count = dict(ngrams_counts)[test_token]
            probability *= token_count / total_char
            if token_count / total_char!=0:
                perplexity+= math.log(token_count / total_char)
    elif n==3:
        # print(test_string_temp)
        test_sentence = test_string_temp.lower()
        test_tokens = test_sentence.split()
        if len(test_tokens) < 3:
            test_tokens = ['<SOS>'] * (3 - len(test_tokens)) + test_tokens
        test_ngrams = list(zip(*[test_tokens[i:] for i in range(3)]))
        for test_ngram in test_ngrams:
            ngram_count3=ngrams_counts3[test_ngram]
            n_1_gram_count3=ngrams_n_1_counts3[test_ngram[:-1]]

            bigram = test_ngram[1:]
            ngram_count2=ngrams_counts2[bigram]
            n_1_gram_count2=ngrams_n_1_counts2[bigram[:-1]]

            unigram=bigram[1:]
            unigram_1=unigram[0]
            # print(test_ngram,"   ",bigram,"   ",unigram_1)
            if unigram_1 not in dict(ngrams_counts1):
                ngram_count1 = 0
            else:
                ngram_count1 = dict(ngrams_counts1)[unigram_1]
            three_gram_prob=0.0
            two_gram_prob=0.0
            one_gram_prob=0.0
            if n_1_gram_count3!=0:
                three_gram_prob = lambdas[2] * ngram_count3 / n_1_gram_count3
            if n_1_gram_count2!=0:
                two_gram_prob = lambdas[1] * ngram_count2 / n_1_gram_count2
            if total_char!=0:
                one_gram_prob = lambdas[0] * ngram_count1 / total_char
            probability *= three_gram_prob + two_gram_prob + one_gram_prob  
            if three_gram_prob + two_gram_prob + one_gram_prob!=0:
                perplexity+= math.log(three_gram_prob + two_gram_prob + one_gram_prob)
    elif n==5:
        test_sentence = test_string_temp.lower()
        test_tokens = test_sentence.split()
        if len(test_tokens) < 5:
            test_tokens = ['<SOS>'] * (5 - len(test_tokens)) + test_tokens
        test_ngrams = list(zip(*[test_tokens[i:] for i in range(5)]))
        # print(test_ngrams)
        for test_ngram in test_ngrams:
            ngram_count5 = ngrams_counts5[test_ngram]
            n_1_gram_count5 = ngrams_n_1_counts5[test_ngram[:-1]]

            ngram4 = test_ngram[1:]
            ngram_count4 = ngrams_counts4[ngram4]
            n_1_gram_count4 = ngrams_n_1_counts4[ngram4[:-1]]

            ngram3 = ngram4[1:]
            ngram_count3 = ngrams_counts3[ngram3]
            n_1_gram_count3 = ngrams_n_1_counts3[ngram3[:-1]]

            ngram2 = ngram3[1:]
            ngram_count2 = ngrams_counts2[ngram2]
            n_1_gram_count2 = ngrams_n_1_counts2[ngram2[:-1]]

            unigram = ngram2[1:]
            unigram_1 = unigram[0]
            if unigram_1 not in dict(ngrams_counts1):
                ngram_count1 = 0
            else:
                ngram_count1 = dict(ngrams_counts1)[unigram_1]
            five_gram_prob=0.0
            four_gram_prob=0.0
            three_gram_prob=0.0
            two_gram_prob=0.0
            one_gram_prob=0.0
            unknown_prob=1/total_char
            if n_1_gram_count5!=0:
                five_gram_prob = lambdas[4] * ngram_count5 / n_1_gram_count5
            if n_1_gram_count4!=0:
                four_gram_prob = lambdas[3] * ngram_count4 / n_1_gram_count4
            if n_1_gram_count3!=0:
                three_gram_prob = lambdas[2] * ngram_count3 / n_1_gram_count3
            if n_1_gram_count2!=0:
                two_gram_prob = lambdas[1] * ngram_count2 / n_1_gram_count2
            if total_char!=0:
                one_gram_prob = lambdas[0] * ngram_count1 / total_char

            probability *= five_gram_prob + four_gram_prob + three_gram_prob + two_gram_prob + one_gram_prob
            if five_gram_prob + four_gram_prob + three_gram_prob + two_gram_prob + one_gram_prob!=0:
                perplexity+= math.log(five_gram_prob + four_gram_prob + three_gram_prob + two_gram_prob + one_gram_prob)
    perplexity = (-1)*perplexity/len(test_ngrams)
    perplexity=np.exp(perplexity)
    return probability,perplexity


def predict_k_words_interpole(test_sentence,corpus_path,n,k):
    all_tokens=find_all_tokens(corpus_path)
    word_probability=set()
    new_file_name = corpus_path.replace('corpus/', '')
    ngrams_counts={}
    ngrams_n_1_counts={}
    ngrams_counts1={}
    ngrams_n_1_counts1={}
    ngrams_counts2={}
    ngrams_n_1_counts2={}
    ngrams_counts3={}
    ngrams_n_1_counts3={}
    ngrams_counts4={}
    ngrams_n_1_counts4={}
    ngrams_counts5={}
    ngrams_n_1_counts5={}
    lambdas=[]
    if n==1:
        ngrams_counts,total_char,vocab_size= unigram_model(corpus_path, new_file_name)
    else:
        ngrams_counts,ngrams_n_1_counts,vocab_size=generalized_model(corpus_path, new_file_name,n)
    if n==3:
        ngrams_counts3,ngrams_n_1_counts3,vocab_size=generalized_model(corpus_path, new_file_name,3)
        ngrams_counts2,ngrams_n_1_counts2,vocab_size=generalized_model(corpus_path, new_file_name,2)
        ngrams_counts1,total_char,vocab_size=unigram_model(corpus_path, new_file_name)
        lambdas = model_train_lambda(new_file_name,n)

    elif n==5:
        ngrams_counts5, ngrams_n_1_counts5, vocab_size = generalized_model(corpus_path, new_file_name, 5)
        ngrams_counts4, ngrams_n_1_counts4, vocab_size = generalized_model(corpus_path, new_file_name, 4)
        ngrams_counts3, ngrams_n_1_counts3, vocab_size = generalized_model(corpus_path, new_file_name, 3)
        ngrams_counts2, ngrams_n_1_counts2, vocab_size = generalized_model(corpus_path, new_file_name, 2)
        ngrams_counts1, total_char, vocab_size = unigram_model(corpus_path, new_file_name)
        lambdas = model_train_lambda(new_file_name, n)
    for word in all_tokens:
        test_string_temp=test_sentence+" "+word
        # print(test_string_temp)
        if n==1:
            probability_temp = interpol_model_smoothing_predict(
                ngrams_counts,ngrams_counts1, ngrams_counts2, ngrams_counts3, ngrams_counts4, ngrams_counts5,
                ngrams_n_1_counts2, ngrams_n_1_counts3, ngrams_n_1_counts4, ngrams_n_1_counts5,
                total_char, vocab_size, corpus_path, n, test_string_temp,lambdas
            )
        else:
            probability_temp = interpol_model_smoothing_predict(
                ngrams_counts,ngrams_counts1, ngrams_counts2, ngrams_counts3, ngrams_counts4, ngrams_counts5,
                ngrams_n_1_counts2, ngrams_n_1_counts3, ngrams_n_1_counts4, ngrams_n_1_counts5,
                total_char, vocab_size, corpus_path, n, test_string_temp,lambdas
            )
        word_probability.add((probability_temp,word))
    sorted_word_probability = sorted(word_probability, key=lambda x: x[0], reverse=True)
    k=int(k)
    i=int(0)
    for word in sorted_word_probability:
        i+=1
        if i>k:
            break
        print(word[1],word[0])

def main():
    if len(sys.argv) != 4:
        print("Usage: generator.py <lm_type> <corpus_path> <k>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k=sys.argv[3]
    print(k)
    n = int(input("Enter the value of n: "))
    test_sentence=input("Enter a Sentence: ")
    # print(test_sentence)

    if lm_type=='l':
        predict_k_words_laplace(test_sentence,corpus_path,n,k)

    elif lm_type=='g':
        predict_k_words_good_turing(test_sentence,corpus_path,n,k)
    
    elif lm_type=='i':
        predict_k_words_interpole(test_sentence,corpus_path,n,k)

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('punkt')
    main()