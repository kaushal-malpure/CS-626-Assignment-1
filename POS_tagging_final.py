# Download the required data (run this only once)
import nltk
nltk.download("brown")
nltk.download("universal_tagset")
nltk.download("punkt")

import random
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import brown
from tqdm.notebook import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
import re
import pickle
import tkinter as tk
from tkinter import messagebox

sns.set_theme()

SEED = 0

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)

setSeed(SEED) # to ensure reproducibility

mean = lambda l : sum(l) / len(l) # computes the mean of a list

def log(x):
    # Used for calculating log probabilities and handles the error when the input is 0
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
    
def plotCM(cm):
    # Plots the confusion matrix provided as a numpy array
    df_cm = pd.DataFrame(cm, index = [t for t in TAGS[1:]], columns = [t for t in TAGS[1:]])
    plt.figure(figsize = (15, 10))
    sns.heatmap(df_cm, annot = True, cmap = plt.cm.Blues)
    plt.show()

def printClasswise(metrics):
    # Pretty-printer for the classwise metrics
    for i in range(N_TAGS - 1):
        print(f"{TAGS[i+1]}: {metrics[i]:.3f}", end = ", ")
    print()

def load():
    # Loads the Brown corpus and converts it into a list
    data = brown.tagged_sents(tagset = "universal") # of type nltk.corpus.reader.util.ConcatenatedCorpusView, each element is a list
    data = list(data) # convert to list for easier processing
    return data

def preprocess(data):
    # Append (^, -) to each data point to indicate the start of a sentence / POS tag sequence
    for s in data:
        s.insert(0, ("^", "-")) # '^' for sentence, '-' for tag

def createFolds(data, k):
    # Randomly shuffles the data and returns k equal sized segments for k-fold cross validation
    n = len(data)
    n_split = n // k # approx split size
    random.shuffle(data)
    folds = []
    for start in range(0, n, n_split):
        end = min(start + n_split, n)
        fold = data[start : end]
        folds.append(fold)
    return folds

def getSentence(d):
    # Given data point d print the sentence contained in it
    s = " ".join([i[0] for i in d])
    return s

def getTags(d):
    # Given data point d return the tag sequence
    t = " ".join([i[1] for i in d])
    return t

def buildTags(data):
    # Compute the set of unique tags present in the data
    tags = set()
    for s in data:
        for w, t in s:
            tags.add(t)
    tags = sorted(list(tags))
    return tags

def buildVocab(data):
    # Compute the set of unique words present in the data, lowercasing is not applied
    vocab = set()
    for s in data:
        for w, t in s:
            vocab.add(w)
    vocab = sorted(list(vocab))
    return vocab

def buildFreqs(data):
    # Computes the frequency matrices for tags, (tag, tag) and (tag, word)

    tf = defaultdict(int) # when the key doesn't exist in the dict, defaultdict automatically initializes it to 0, unlike normal dicts
    ef = defaultdict(int)
    tag_freq = defaultdict(int)

    for s in data:
        p = "-" # the initial value of the previous tag (this acts as the tag for ^: the sentence start)
        for w, t in s:
            tf[(p, t)] += 1 # auto-init to 0 when key doesn't exit
            ef[(t, w)] += 1 # each word has a corresponding tag
            tag_freq[t] += 1
            p = t # update prev tag

    return tf, ef, tag_freq

def buildMatrices(tf, ef, tag_freq):
    # Builds the transition and emmission matrices from the frequency matrices while applying smoothing 

    # Initialize
    tm = np.zeros((N_TAGS, N_TAGS))
    em = np.zeros((N_TAGS, N_VOCAB))

    alpha = 1e-8 # handles values when certain combinations of tags / bigrams don't occur in the data
    # Reference for this: http://ivan-titov.org/teaching/nlmi-15/lecture-4.pdf

    for i in range(N_TAGS):

        # Build the transition matrix
        for j in range(N_TAGS):
            ti = id2tag(i)
            tj = id2tag(j)
            tm[i, j] = (tf[(ti, tj)] + alpha) / (tag_freq[ti] + alpha*N_TAGS)

        # Build the mmission matrix
        for j in range(N_VOCAB):
            ti = id2tag(i)
            wj = id2word(j)
            em[i, j] = (ef[(ti, wj)] + alpha) / (tag_freq[ti] + alpha*N_VOCAB)

    return tm, em

def train(data):
    # Just runs the past two functions one after the other and returns the transition and emmission matrices
    tf, ef, tag_freq = buildFreqs(data)
    tm, em = buildMatrices(tf, ef, tag_freq)
    return tm, em

def viterbi(tm, em, s, word2id):
    # Performs viterbi decoding on sentence s, which is assumed to be a list of words starting from ^
    # Follows the pseudo-code given in the slides

    n_sent = len(s) # length of the sentence to be decoded

    # Create tables to be used for dynamic programming
    best_probs = np.zeros((n_sent, N_TAGS)) # stores the log probabilities to ensure values don't become too small by repeated multiplications later
    best_tags  = np.zeros((n_sent, N_TAGS))

    # Initialize the first row of best_probs
    for i in range(N_TAGS):
        best_probs[0, i] = log(tm[0, i]) + log(em[i, word2id(s[0])])

    # This can be thought of as a "forward pass"
    for i in range(1, n_sent):

        w_prev = s[i - 1]
        w = s[i]

        for j in range(N_TAGS):
            pmax = -np.inf
            best_tag = None # stores index of the best tag for the current word
            for k in range(N_TAGS):        
                p = best_probs[i - 1, k] + log(tm[k, j])
                if p > pmax:
                    pmax = p
                    best_tag = k

            try:
                wid = word2id(w)
            except KeyError: # handles KeyError
                wid = -1 # word not in vocab

            if wid != -1:
                best_probs[i, j] = pmax + log(em[j, wid]) # the second term is constant when taking argmax over k
            else:
                # In this case the bigram doesn't occur in the training data
                best_probs[i, j] = pmax + log(em.min()) # assign the minimum emmission probability to this tag

            best_tags[i, j] = best_tag

    pred = [] # the tag predictions for the sentence

    # Get the last tag 
    pmax = -np.inf
    end_tag = None
    for i in range(N_TAGS):
        if best_probs[-1, i] > pmax:
            pmax = best_probs[-1, i]
            end_tag = i
    pred.append(end_tag)

    # Now backtrack to the start following the path which led to end_tag
    # This can be thought of as a "backward pass"
    t = end_tag # init
    for i in range(n_sent - 1, 1, -1):
        t = int(best_tags[i, t])
        pred.append(t)

    pred.reverse() # as we have built this starting from the back

    return pred

def evaluate(data, tm, em):
    # Evaluates the trained HMM model on the data

    tags_true = []
    tags_pred = []

    for d in tqdm(data):
        s = [i[0] for i in d]
        t_true = [tag2id(i[1]) for i in d[1:]] # ignoring "-" at the start
        t_pred = viterbi(tm, em, s)
        tags_true += t_true
        tags_pred += t_pred
        
    return tags_true, tags_pred

# Load the required data
data = load()
preprocess(data)

N = len(data)
print(f"Length: {N:,}")

TAGS = buildTags(data)
N_TAGS = len(TAGS)
print(f"Tagset size: {N_TAGS} (includes the dummy '-' tag)")
print(TAGS)

# Maps
id2tag = lambda i : TAGS[i]
tagmap = {TAGS[i]:i for i in range(N_TAGS)}
tag2id = lambda i : tagmap[i]

# Print some random statements and the corresponding tags
for i in random.sample(range(N), 5):
    print(getSentence(data[i]))
    print(getTags(data[i]))
    print()

folds = createFolds(data, k = 5)

# Initialize the metrics

# Avg cm
avg_cm = 0 # ignoring the dummy tag added at the start of each sentence

# Initialize average metrics
avg_acc = 0
avg_prec = 0
avg_rec = 0
avg_f1 = 0
avg_f05 = 0
avg_f2 = 0

# Average tagwise metrics
avg_prec_tagwise = 0
avg_rec_tagwise = 0
avg_f1_tagwise = 0

# Start cross-validation
tm_list = []
em_list = []

flag = input("------------------------\nINPUT 1 for 5 FOLD VALIDATION 0 FOR TESTING")
if flag == "1":
    for fv in range(5):
        print("--------------------------------------------")
        print(f"---------------FOLD {fv} ------------------")
        print("--------------------------------------------")
        # Separate out the validation fold
        val_data = folds[fv]

        # Concatenate the training folds
        train_data = []
        for ft in range(5):
            if ft != fv:
                train_data += folds[ft]
        with open(f'train_data_{fv}.pkl','wb') as train_file:
            pickle.dump(train_data, train_file)
        # Update vocab based on training data (these variables are used globally)
        VOCAB = buildVocab(train_data)
        N_VOCAB = len(VOCAB)
        id2word = lambda i : VOCAB[i]
        wordmap = {VOCAB[i]:i for i in range(N_VOCAB)}
        word2id = lambda i : wordmap[i]

        # Train
        tm, em = train(train_data)

        # Save to file
        with open(f'tm_file_{fv}.pkl', 'wb') as tm_file:
            pickle.dump(tm, tm_file)

        with open(f'em_file_{fv}.pkl', 'wb') as em_file:
            pickle.dump(em, em_file)

        # Validate
        tags_true, tags_pred = evaluate(val_data, tm, em)

        # Compute metrics

        metric_labels = [i+1 for i in range(N_TAGS - 1)] # ignore the first dummy tag '-'

        cm = confusion_matrix(tags_true, tags_pred, normalize = "true", labels = metric_labels)
        acc = accuracy_score(tags_true, tags_pred)

        prec = precision_score(tags_true, tags_pred, average = None, labels = metric_labels)
        rec = recall_score(tags_true, tags_pred, average = None, labels = metric_labels)
        f1 = f1_score(tags_true, tags_pred, average = None, labels = metric_labels)

        f05 = fbeta_score(tags_true, tags_pred, beta = 0.5, average = "macro", labels = metric_labels)
        f2 = fbeta_score(tags_true, tags_pred, beta = 2, average = "macro", labels = metric_labels)

        # Tag-wise metrics
        print("Tag-wise precision:")
        printClasswise(prec)

        print("Tag-wise recall:")
        printClasswise(rec)

        print("Tag-wise F1:")
        printClasswise(f1)

        # Overall metrics for this fold
        print("Accuracy:", acc)
        print("Avg. precision:", mean(prec))
        print("Avg. recall:", mean(rec))
        print("Avg. f_0.5:", f05)
        print("Avg. f1:", mean(f1))
        print("Avg. f2:", f2)

        # plotCM(cm)

        # Update global avg values
        avg_acc += acc
        avg_prec += mean(prec)
        avg_rec += mean(rec)
        avg_f1 += mean(f1)
        avg_f05 += f05
        avg_f2 += f2

        # Update tagwise metrics
        avg_prec_tagwise += prec
        avg_rec_tagwise += rec
        avg_f1_tagwise += f1

        # Update cm
        avg_cm += cm

        tm_list.append(tm)
        em_list.append(em)

    # Average over the number of folds

    avg_cm /= 5

    avg_acc /= 5
    avg_prec /= 5
    avg_rec /= 5
    avg_f1 /= 5
    avg_f05 /= 5
    avg_f2 /= 5

    avg_prec_tagwise /= 5
    avg_rec_tagwise /= 5
    avg_f1_tagwise /= 5

    # Final metrics reported

    # Tag-wise metrics
    print("Tag-wise precision:")
    printClasswise(avg_prec_tagwise)

    print("Tag-wise recall:")
    printClasswise(avg_rec_tagwise)

    print("Tag-wise F1:")
    printClasswise(avg_f1_tagwise)

    # Overall metrics for this fold
    print("Accuracy:", avg_acc)
    print("Avg. precision:", avg_prec)
    print("Avg. recall:", avg_rec)
    print("Avg. f_0.5:", avg_f05)
    print("Avg. f1:", avg_f1)
    print("Avg. f2:", avg_f2)

    # plotCM(avg_cm)

# else :
#     t = "1"

#     # Load from file
#     with open('tm_file_0.pkl', 'rb') as tm_file:
#         loaded_tm = pickle.load(tm_file)

#     with open('em_file_0.pkl', 'rb') as em_file:
#         loaded_em = pickle.load(em_file)



t = "1"


# while t == "1":
def POS_tagger():
    # flag = input("\nDo you want to enter a sentence? (YES = 1, NO = 0)\n")
    # sentence = input("\nEnter a sentence:\n")
    sentence = sentence_entry.get()
    sentence = re.findall(r"[\w']+|[.,!?;]", sentence)
    sentence.insert(0, '^')

    with open('train_data_0.pkl', 'rb') as train_file:
        train_data = pickle.load(train_file)

    VOCAB = buildVocab(train_data)
    N_VOCAB = len(VOCAB)
    id2word = lambda i : VOCAB[i]
    wordmap = {VOCAB[i]:i for i in range(N_VOCAB)}
    word2id = lambda i : wordmap[i]

    # Load from file
    with open('tm_file_0.pkl', 'rb') as tm_file:
        loaded_tm = pickle.load(tm_file)

    with open('em_file_0.pkl', 'rb') as em_file:
        loaded_em = pickle.load(em_file)

    tags = viterbi(loaded_tm, loaded_em, sentence, word2id)
    final_tags = ""
    for tag in tags:
        final_tags = final_tags + id2tag(tag) + " "
    # print(final_tags)
    result_label.config(text=final_tags)

    # return flag

# while t=="1":
#     t = POS_tagger(t)

# def function_returner():
#     with open('train_data_0.pkl', 'rb') as train_file:
#         train_data = pickle.load(train_file)

#     VOCAB = buildVocab(train_data)
#     N_VOCAB = len(VOCAB)
#     id2word = lambda i : VOCAB[i]
#     wordmap = {VOCAB[i]:i for i in range(N_VOCAB)}
#     word2id = lambda i : wordmap[i]

#     return id2word, wordmap, word2id
#     pass

# def POS_tagger():
#     sentence = input("\nEnter a sentence:\n")
#     sentence = re.findall(r"[\w']+|[.,!?;]", sentence)
#     sentence.insert(0, '^')


# Set up the GUI window
root = tk.Tk()
root.title("POS Tagging GUI")

# Input field for sentence
tk.Label(root, text="Enter a sentence:").grid(row=0, column=0, padx=10, pady=10)
sentence_entry = tk.Entry(root, width=50)
sentence_entry.grid(row=0, column=1, padx=10, pady=10)

# Button to trigger POS tagging
tag_button = tk.Button(root, text="Tag Sentence", command=POS_tagger)
tag_button.grid(row=1, column=0, columnspan=2, pady=10)

# Label to display the POS tagging results
result_label = tk.Label(root, text="POS Tags will be shown here", justify="left", anchor="w")
result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()