import nltk
from nltk import word_tokenize,sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def tokenize(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return words
sententce = input("Your sentence: ")
print(f"Tokenized Sentence: {tokenize(sententce)}")

