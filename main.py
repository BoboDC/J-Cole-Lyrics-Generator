import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Dropout, CuDNNLSTM, Bidirectional
from keras.models import Sequential
from keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adamax


data = pd.read_fwf('data/j cole.txt', header=None)
xData = data[0]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(xData.astype(str).str.lower())
totalWords = len(tokenizer.word_index)+1
tokenVerse = tokenizer.texts_to_sequences(xData.astype(str))
allWordsCount = tokenizer.word_counts

inputData = list()
for i in tokenVerse:
    for t in range(1, len(i)):
        seq = i[:t+1]
        inputData.append(seq)
        
maxVerse = max([len(x) for x in inputData]) 
inputVerse = np.array(pad_sequences(inputData, maxlen=maxVerse, padding='pre'))

x = inputVerse[:,:-1]
labels = inputVerse[:,-1]

y = tf.keras.utils.to_categorical(labels, num_classes=totalWords)

def createModel():
    model = Sequential()
    model.add(Embedding(totalWords, 40, input_length=maxVerse-1))
    model.add(Bidirectional(CuDNNLSTM(256)))
    model.add(Dropout(0.1))
    model.add(Dense(totalWords, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.01), metrics=['accuracy'])
    
    model.fit(x, y, batch_size=32, epochs=500, validation_split = 0.2)
    model.save('model.h5')

# createModel()
model = load_model('model.h5')
model.summary()
def songCreator(input_text, next_words, space):
    aux = 0
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=maxVerse-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word
        aux += 1
        if aux % space == 0:
            input_text += '\n'
    return input_text

randomBars = xData[np.random.randint(11222)]
randomBars += '\n'
spaceBars = randomBars.split()
song = songCreator(randomBars, 5*len(spaceBars), space=len(spaceBars))
print(song)
    