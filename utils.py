import pickle
import unicodedata
import nltk
import torch
from torch import lstm_cell, nn
import time
import random
import numpy as np
import pickle as pkl
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed , Input ,Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

########################################################################################
# Read the letters from the pickle files which we will use
def get_letters():
    file_path = 'constants/arabic_letters.pickle'
    with open(file_path, 'rb') as file:
        letters = pickle.load(file)
    return letters
########################################################################################
# Read the diacritics from the pickle files which we will use
def get_diacritics():
    file_path = 'constants/diacritics.pickle'
    with open(file_path, 'rb') as file:
        diacritics = pickle.load(file)

    return diacritics
########################################################################################
# Read the diacritics from the pickle files which we will use
def get_diacritics2id():

    file_path = 'constants/diacritic2id.pickle'
    with open(file_path, 'rb') as file:
        diacritics2id = pickle.load(file)

    return diacritics2id

########################################################################################
# Read TRAINING dataset given
def read_training_dataset(file_path = "dataset/train.txt"):
    training_sentences = []
    with open(file_path, "r", encoding="utf-8") as file:
        # Read each line in the file
        for line in file:
            # Strip any leading or trailing whitespace from the line
            line = line.strip()
            # Add the line to the list
            training_sentences.append(line)
    if(len(training_sentences)==50000):
        print("Read training set successfully")
        return training_sentences
    
########################################################################################
# Read DEV dataset given
def read_dev_dataset(file_path = "dataset/val.txt"):
    dev = []
    with open(file_path, "r", encoding="utf-8") as file:
        # Read each line in the file
        for line in file:
            line = line.strip()
            dev.append(line)
    #print(len(dev))
    if(len(dev)==2500):
        print("Read validation set successfully")
        return dev
    
########################################################################################

def separate_word_to_letters_diacritics(arabic_text):
    # Normalize the text to handle different Unicode representations
    normalized_text = unicodedata.normalize('NFKD', arabic_text)
    letters =[]
    diacritics=[]
    ind=0
    while ind < len(normalized_text):
        temp=[]
        if not unicodedata.combining(normalized_text[ind]):
            letters.append(normalized_text[ind])
            # print("added to letters",normalized_text[ind])

            if(ind+1 < len(normalized_text) and not unicodedata.combining(normalized_text[ind+1])):
                diacritics.append(temp)
                # print("added to diacritics from 1st",temp)
            ind+=1

        else:
            while ind < len(normalized_text) and unicodedata.combining(normalized_text[ind]):
                # diacritics.pop(0)
                temp.append(normalized_text[ind])
                ind+=1
            diacritics.append(temp)
            # print("added to diacritics",temp)
    
    return letters, diacritics

########################################################################################
def tokenize_to_vocab(data, vocab):
    tokenized_sentences_word, tokenized_sentences_letters, tokenized_sentences_diacritics = [], [],[]

    for d in (data):
            tokens = nltk.word_tokenize(d, language="arabic", preserve_line=True)
            # Add the start sentence <s> and end sentence </s> tokens at the beginning and end of each tokenized sentence
            tokens.insert(0,"<s>")
            tokens.append("</s>")

            vocab.update(tokens)

            word_letters=[]
            word_diacritics=[]
            for token in (tokens):
                if token != "<s>" and token != "</s>":
                    # letters = separate_arabic_to_letters(token)
                    letter, diacritic = separate_word_to_letters_diacritics(token)
                    word_diacritics.append(diacritic)
                    word_letters.append(letter)
                else:
                    word_letters.append(token)
                    word_diacritics.append(token)

            tokenized_sentences_letters.append(word_letters)
            tokenized_sentences_diacritics.append(word_diacritics)
            tokenized_sentences_word.append(tokens)  
              
    return vocab, tokenized_sentences_word,tokenized_sentences_letters,tokenized_sentences_diacritics

########################################################################################
def map_data(data_raw):
    pass

########################################################################################
def create_model(CHARACTERS_MAPPING, CLASSES_MAPPING ):
    ''' Creates diacritization model '''


    SelectedLSTM = LSTM
    
    inputs = Input(shape=(None,))
    
    embeddings = Embedding(input_dim=len(CHARACTERS_MAPPING),
                           output_dim=25,
                           embeddings_initializer=glorot_normal(seed=961))(inputs)
    
    blstm1 = Bidirectional(SelectedLSTM(units=256,
                                     return_sequences=True,
                                     kernel_initializer=glorot_normal(seed=961)))(embeddings)
    dropout1 = Dropout(0.5)(blstm1)
    blstm2 = Bidirectional(SelectedLSTM(units=256,
                                     return_sequences=True,
                                     kernel_initializer=glorot_normal(seed=961)))(dropout1)
    dropout2 = Dropout(0.5)(blstm2)
    
    dense1 = TimeDistributed(Dense(units=512,
                                   activation='relu',
                                   kernel_initializer=glorot_normal(seed=961)))(dropout2)
    dense2 = TimeDistributed(Dense(units=512,
                                   activation='relu',
                                   kernel_initializer=glorot_normal(seed=961)))(dense1)
    
    output = TimeDistributed(Dense(units=len(CLASSES_MAPPING),
                                   activation='softmax',
                                   kernel_initializer=glorot_normal(seed=961)))(dense2)
    
    model = Model(inputs, output)
    
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    
    return model
########################################################################################
class DataGenerator(Sequence):
    ''' Costumized data generator to input line batches into the model '''
    def __init__(self, lines, batch_size):
        self.lines = lines
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, Y_batch = map_data(lines)
        
        X_max_seq_len = np.max([len(x) for x in X_batch])
        Y_max_seq_len = np.max([len(y) for y in Y_batch])
        
        assert(X_max_seq_len == Y_max_seq_len)
        
        X = list()
        for x in X_batch:
            x = list(x)
            x.extend([CHARACTERS_MAPPING['<PAD>']] * (X_max_seq_len - len(x)))
            X.append(np.asarray(x))
        
        Y_tmp = list()
        for y in Y_batch:
            y_new = list(y)
            y_new.extend(to_categorical([CLASSES_MAPPING['<PAD>']] * (Y_max_seq_len - len(y)), len(CLASSES_MAPPING)))
            Y_tmp.append(np.asarray(y_new))
        Y_batch = Y_tmp
        
        Y_batch = np.array(Y_batch)
        
        return (np.array(X), Y_batch)
    
########################################################################################
def fit_model(model, epochs, batch_size, train_split,val_split):
    ''' Fits model '''

    
    # create training and validation generators
    training_generator = DataGenerator(train_split, batch_size)
    val_generator = DataGenerator(val_split, batch_size)
    

    # fit model
    history = model.fit(x=training_generator,
              validation_data=val_generator,
              epochs=epochs
              )

    # return history
    return history
########################################################################################
def predict(line, model):
    ''' predict test line '''
    X, _ = map_data([line])
    predictions = model.predict(X).squeeze()
    # get most probable diacritizations for each character
    predictions = predictions[1:]
    
    # initialize empty output line 
    output = ''
    # loop on input characters and predicted diacritizations
    for char, prediction in zip(remove_diacritics(line), predictions):
        # append character
        output += char
        # if character is not an arabic letter continue
        if char not in ARABIC_LETTERS_LIST:
            continue
        
        if '<' in REV_CLASSES_MAPPING[np.argmax(prediction)]:
            continue

        # if character in arabic letters append predicted diacritization
        output += REV_CLASSES_MAPPING[np.argmax(prediction)]

    return output

########################################################################################

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Transform input to embedding
        x = self.embedding(x)

        # Initialize hidden state
        hidden = self.init_hidden(batch_size)

        # Pass the embedded input and hidden state through the RNN
        out, _ = self.rnn(x, hidden)

        # Pass the output of the RNN through a fully connected layer
        out = self.fc(out[:, -1, :])

        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden

# Example usage:
# model = RNNModel(input_size=36, hidden_size=128, output_size=10, n_layers=2)