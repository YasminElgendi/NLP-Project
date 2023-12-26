### Add all imports
# import subprocess
import re
import nltk
from nltk.stem.isri import ISRIStemmer
import pyarabic.trans
from pyarabic.araby import strip_diacritics
import numpy as np
#######################################################

# Fat7a = 4, damma =5, kasra = 6, sokoon = 7
# tanween fat7a =1, damma=2 , kasraa=3
# shadda =70
# CONSTANTS
DIACRITICS = np.array([1,2,3,4,5,6,7,70])

def read_training_dataset(file_path = "dataset/train.txt", encoding = "utf-8"):
    training_sentences = []
    with open(file_path, "r", encoding=encoding) as file:
        # Read each line in the file
        for line in file:
            # Strip any leading or trailing whitespace from the line
            line = line.strip()
            # Add the line to the list
            training_sentences.append(line)
    if(len(training_sentences)==50000):
        print("Read training set successfully")
        return training_sentences

def read_dev_dataset(file_path = "dataset/val.txt", encoding = "utf-8"):
    dev = []
    with open(file_path, "r", encoding=encoding) as file:
        # Read each line in the file
        for line in file:
            line = line.strip()
            dev.append(line)
    #print(len(dev))
    if(len(dev)==2500):
        print("Read validation set successfully")
        return dev

def data_Cleaning():
    pass


def stem(word):
    stemmer = ISRIStemmer()
    stemmed_word = stemmer.stem(word)
    return stemmed_word


    

def tokenize_to_vocab(data, vocab):
    tokenized_sentences = []
    for d in (data):
            tokens = nltk.word_tokenize(d, language="arabic", preserve_line=True)
            # Add the start sentence <s> and end sentence </s> tokens at the beginning and end of each tokenized sentence
            # for token in tokens:
            #     token = stem(token)
            tokens.insert(0,"<s>")
            tokens.append("</s>")
            vocab.update(tokens)
            tokenized_sentences.append(tokens)
    
    return vocab, tokenized_sentences

def extract_arabic_letters(input_word):
    # Define a regular expression pattern for standard Arabic letters
    # pattern = re.compile(r"[\u0600-\u06FF]+") #NOTE returns all letter and dialects
    pattern = re.compile(r"[\u0621-\u064A\s]+")

    # Find all matches and join them
    # This will exclude non-letter characters and diacritics
    result = "".join(pattern.findall(input_word))

    return result

def extract_diacritics(word, coding = "decimal"):
    # Return encoding of word diacritics in decimal
    return pyarabic.trans.encode_tashkeel(word, coding)[1]

def join_word_diacritics(word, diacritics, coding = "decimal"):
    # Writes a word with its corresponding diacritics
    return pyarabic.trans.decode_tashkeel(word, diacritics, coding)

def word_to_embedding(word):
    embedding = [int(ord(x)) for x in word]
    return sum(embedding)

# def char_to_embedding(char):
#     embedding = [int(ord(x)) for x in word]
#     return sum(embedding)