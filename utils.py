import pickle
import unicodedata
import nltk

########################################################################################
# Read the letters and diacritics from the pickle files which we will use
def get_letters():

    file_path = 'arabic_letters.pickle'
    with open(file_path, 'rb', encoding="utf-8") as file:
        letters = pickle.load(file)

    return letters

def get_diacritics2id():

    file_path = 'diacritics2id.pickle'
    with open(file_path, 'rb', encoding="utf-8") as file:
        diacritics2id = pickle.load(file)

    return diacritics2id

########################################################################################

# Read datasets given
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
def separate_arabic_to_letters(arabic_text):
    # Normalize the text to handle different Unicode representations
    normalized_text = unicodedata.normalize('NFKD', arabic_text)
    
    # Separate the normalized text into individual letters
    letters = [char for char in normalized_text if not unicodedata.combining(char)]
    
    return letters

########################################################################################
def tokenize_to_vocab(data, vocab):
    tokenized_sentences_word, tokenized_sentences_letters = [], []

    for d in (data):
            tokens = nltk.word_tokenize(d, language="arabic", preserve_line=True)
            # Add the start sentence <s> and end sentence </s> tokens at the beginning and end of each tokenized sentence
            tokens.insert(0,"<s>")
            tokens.append("</s>")

            vocab.update(tokens)

            word_letters=[]
            for token in (tokens):
                if token != "<s>" and token != "</s>":
                    # letters = separate_arabic_to_letters(token)
                    word_letters.append(separate_arabic_to_letters(token))
                else:
                    word_letters.append(token)

            tokenized_sentences_letters.append(word_letters)
            tokenized_sentences_word.append(tokens)  
              
    return vocab, tokenized_sentences_word,tokenized_sentences_letters

def tokenize_to_letters(tokenized_sentences):
    word_letters=[]
    for tokenized_sentence in tokenized_sentences:
        for token in (tokenized_sentence):
            if token != "<s>" and token != "</s>":
                # letters = separate_arabic_to_letters(token)
                word_letters.append(separate_arabic_to_letters(token))
    return word_letters