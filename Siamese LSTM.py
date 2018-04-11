import keras
import pandas as pd
import nltk
import string
from keras.layers import *
from keras.preprocessing.sequence import *
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
TBT = nltk.tokenize.TreebankWordTokenizer()
LEM = nltk.stem.WordNetLemmatizer()
p_stemmer = nltk.stem.PorterStemmer()
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 10
import matplotlib.pyplot as plt
dict = {}
def read_dataset():
    train = pd.read_csv("train.csv")
    train_y = train["is_duplicate"]

    train_x_q1 = train["question1"]
    train_x_q2 = train["question2"]

    train_x_q1 = preprocess(train_x_q1)
    train_x_q2 = preprocess(train_x_q2)

    train_x_q1 = np.asarray(train_x_q1)
    train_x_q2 = np.asarray(train_x_q2)
    train_y = np.asarray(train_y)
    return train_x_q1,train_x_q2,train_y

def read_test_dataset():
    test = pd.read_csv("test.csv")
    test_id = test["test_id"]
    test_x_q1 = test["question1"]
    test_x_q2 = test["question2"]

    test_x_q1 = preprocess(test_x_q1)
    test_x_q2 = preprocess(test_x_q2)

    test_x_q1 = np.asarray(test_x_q1)
    test_x_q2 = np.asarray(test_x_q2)
    test_id = np.asarray(test_id)

    return test_x_q1,test_x_q2,test_id

def preprocess(data):
    stop = stopwords.words('english')
    stop += list(string.punctuation)
    #stop = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    list_of_X = data.apply(lambda row: str(row).lower())
    list_of_X = list_of_X.apply(lambda row: TBT.tokenize(row))
    list_of_X = list_of_X.apply(lambda row: [LEM.lemmatize(i) for i in row])
    #print(np.asarray(list_of_X)[3])
    list_of_X = list_of_X.apply(lambda row: [p_stemmer.stem(i) for i in row])
    #print(np.asarray(list_of_X)[3])
    list_of_X = list_of_X.apply(lambda row: [i for i in row if i not in stop])
    list_of_X = list_of_X.apply(lambda row: str(row))
    # for i in range(len(data)):
    #     print(np.asarray(data)[i])
    #     print(np.asarray(list_of_X)[i])
    return list_of_X

def embedding(word_index):
    embeddings_index = {}
    f = open(os.path.join('glove.6B.50d.txt'))
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    nb_words = len(word_index)
    word_embedding_matrix = np.random.randn(nb_words + 1, EMBEDDING_DIM)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
    return word_embedding_matrix

def seq_data(data,word_to_ind):
    temp = data.apply(lambda row: [word_to_ind[i] for i in row])
    return temp

data_Q1 , data_Q2 ,label = read_dataset()

# test_Q1 , test_Q2,test_id = read_test_dataset()

tokenizer = Tokenizer(num_words=200000)
# tokenizer.fit_on_texts(data_Q1+data_Q2+test_Q1+test_Q2)
tokenizer.fit_on_texts(data_Q1+data_Q2)
question1_word_sequences = tokenizer.texts_to_sequences(data_Q1)
question2_word_sequences = tokenizer.texts_to_sequences(data_Q2)
word_index = tokenizer.word_index
word_embedding_matrix = embedding(word_index)
data_Q1 = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='pre',truncating='post')
data_Q2 = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='pre',truncating='post')
datax = np.stack((data_Q1, data_Q2), axis=1)
X_train, X_test, y_train, y_test = train_test_split(datax, label, test_size=0.2, random_state=50)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]
def train_model():

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    Q1 = Embedding(input_dim=len(word_index)+1,output_dim=EMBEDDING_DIM,
                            weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH)(question1)

    q1 = GRU(256,unroll=True)(Q1)


    Q2 = Embedding(input_dim=len(word_index)+1, output_dim=EMBEDDING_DIM,
                            weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH)(question2)

    q2 = GRU(256,unroll=True)(Q2)

    q1q2 = concatenate([q1,q2])

    q1q2 = Dense(200, activation='tanh')(q1q2)

    is_duplicate = Dense(1, activation='sigmoid')(q1q2)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])

    history = model.fit([Q1_train, Q2_train],
                        y_train,
                        epochs=10,
                        verbose=1,
                        validation_data=([Q1_test,Q2_test],y_test),
                        batch_size=128)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model
# def test_model(model):
#     question1_word_sequences_test = tokenizer.texts_to_sequences(test_Q1)
#     question2_word_sequences_test = tokenizer.texts_to_sequences(test_Q2)
#
#     test_Q1 = pad_sequences(question1_word_sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='post')
#     test_Q2 = pad_sequences(question2_word_sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='post')
#
#     predicted = model.predict([test_Q1, test_Q2])
#     print(predicted)
#     return predicted
model = train_model()
# predicted = test_model(model)




