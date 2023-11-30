### Add all imports
import subprocess
import arabic_reshaper
from bidi.algorithm import get_display
#######################################################

def install_libraries(requirements_file_path):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file_path])

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

def tokenize(sentence):
    # Reshape the Arabic sentence to its proper form
    reshaped_text = arabic_reshaper.reshape(sentence)
    
    # Get the sentence with proper right-to-left ordering
    displayed_text = get_display(reshaped_text)
    
    # Split the sentence into tokens (words)
    tokens = displayed_text.split()
    
    return tokens