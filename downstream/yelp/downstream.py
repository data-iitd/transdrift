
from keras.datasets import imdb
import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras_preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.optimizers import Adam
import os
from keras.layers import Bidirectional
import arch_downstream as ARCH
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from keras.callbacks import TensorBoard
import tensorflow as tf



data_file = ARCH.data_file
data = pd.read_csv(data_file)

data['text'] = data['text'].str.lower()



# data['select'] = np.array(list(map(int, data['date'].str[:4])))>2015



# data = data[data.select].copy()



# data.drop(data.index[20000:], inplace=True)




stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ]

# Cleaning reviews
def remove_stopwords(data):
    data['clean_review'] = data['text'].apply(lambda x : ' '.join([word for word in str(x).split() if word not in (stopwords)]))
    return data

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result
    
data_without_stopwords = remove_stopwords(data)
data_without_stopwords['clean_review'] = data_without_stopwords['text'].apply(lambda cw : remove_tags(cw))
data_without_stopwords['clean_review'] = data_without_stopwords['text'].str.replace('[{}]'.format(string.punctuation), ' ')


# data_without_stopwords = data_without_stopwords.drop(columns = ['review_id', 'user_id', 'business_id', 'useful', 'funny', 'cool', 'text', 'date', 'select', 'Unnamed: 0'])
data_without_stopwords.reset_index(drop=True, inplace=True)
data_without_stopwords

reviews = data_without_stopwords['clean_review']
reviews_list = []

for i in range(len(reviews)):
    reviews_list.append(reviews[i])
stars = data_without_stopwords['stars']

y = np.array(stars)

# This y will be the label of classification task.
y = np.array(list(map(int, y)))
y_ = np.zeros((len(y), 5))
for i in range(0, len(y)):
    y_[i][y[i]-1] = 1
y=y_

X_train, X_test,Y_train, Y_test = train_test_split(reviews_list, y, test_size=0.2, random_state = 45)

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(X_train)
words_to_index = tokenizer.word_index

# Reading word embedding of common words.
def read_vector(inp_vec):
    Dict = open(inp_vec, 'r')
    lines = Dict.readlines()
    word_to_vec_map = {}
    common_words = open(ARCH.common_words, 'r')
    comn_words = common_words.read()
    comn_wrds = comn_words.split("\n")
    for line in lines:
        wrd_vec = line.split(" ")
        key = wrd_vec[0]
        if(key not in comn_wrds):
            continue
        wrd_vec = wrd_vec[1:]
        wrd_vec[-1] = wrd_vec[-1][:-1]
        word_to_vec_map[key] = np.array(list(map(np.float64,wrd_vec)))
    Dict.close()
    common_words.close()
    return word_to_vec_map

def rating_model(input_shape):
    deep_inputs = Input(input_shape)
    embeddings = embedding_layer(deep_inputs)
    LSTM_Layer_1 = LSTM(128)(embeddings)
    dense_layer_1 = Dense(5, activation='sigmoid')(LSTM_Layer_1)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)
    return model





def add_score_predictions(data, reviews_list_idx, model):
    data['sentiment score'] = 0
    reviews_list_idx = pad_sequences(reviews_list_idx, maxlen=maxLen, padding='post')
    review_preds = model.predict(reviews_list_idx)
    data['sentiment score'] = review_preds
    pred_sentiment = np.array(list(map(lambda x : np.where(x == np.amax(x))[0]+1, review_preds)))
    data['predicted sentiment'] = 0
    data['predicted sentiment'] = pred_sentiment
    return data

cvscores = []
flag = True

# Running downstream task using 100 word_embedding files and taking average for calculating accuracies.
for i in range(1,101):
    log_dir = os.path.join(ARCH.log_path, datetime.today().isoformat() , ARCH.emb_name + "_" + data_file[:-4] + "_test_set_" + str(i))
    pred_idx = str(i)
    os.makedirs(log_dir, exist_ok=True)
 
    # writer = SummaryWriter(log_dir)
    writer = TensorBoard(log_dir=log_dir)
    
    inp_file = ARCH.inp_dir + "/" + ARCH.emb_name + pred_idx
    word_to_vec_map = read_vector(inp_file + '.txt')
    maxLen = 150
    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['i'].shape[0]

    emb_matrix = np.zeros((vocab_len, embed_vector_len))
    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index-1, :] = embedding_vector

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False)
    X_train_indices = tokenizer.texts_to_sequences(X_train)
    X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
    
    model = rating_model((maxLen,))
    adam = Adam(learning_rate = ARCH.learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    model.fit(X_train_indices, Y_train, batch_size=ARCH.batch_size, epochs=ARCH.epochs, validation_split=0.2, callbacks = [writer])
    
    X_test_indices = tokenizer.texts_to_sequences(X_test)
    X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')
    
    model.evaluate(X_test_indices, Y_test)
    
    preds = model.predict(X_test_indices)
    
    reviews_list_idx = tokenizer.texts_to_sequences(reviews_list)
    data = add_score_predictions(data, reviews_list_idx, model)
    data2 = data.iloc[16000:]
    result_bool = data2["stars"] == data2["predicted sentiment"]
    count = 0
    for j in result_bool:
        if(j):
            count += 1
    s_core = count/len(result_bool)
    if(flag):
        f = open(ARCH.log_path + "/" + ARCH.emb_name + "_" + data_file[:-4] + "_test_accuracy.txt", "w")
        flag = False
    f.write("accuracy" + str(i) + ": " + str(s_core) + "\n")
    cvscores.append(s_core)
    
    try:
        out_dir = ARCH.out_dir
        os.mkdir(out_dir)
    except:
        pass
#     data2.to_csv(out_dir + ARCH.out_file + pred_idx + "_" + ARCH.d2_small + "_" + data_file)
    

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
f.write("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
f.close()






