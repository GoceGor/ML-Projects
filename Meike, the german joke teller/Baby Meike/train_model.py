from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

##############################################################
# Project configuration

NUM_OF_STEPS_PER_EPOCH = 100
NUM_OF_EPOCHS = 50
batch_size = 256
NUM_OF_CELLS = 128
LEARNING_RATE = 0.01
MAX_JOKE_LENGTH = 200 # 200 characters
###############################################################
###############################################################
###############################################################
###############################################################
#
#           Helper functions :
#
###############################################################
################################################################
#################################################################
# Batch generator

def batch_generator(batch_size):
    
    while True:
        x = np.zeros((batch_size, maxjoke-1, len(chars)), dtype=np.bool)
        y = np.zeros((batch_size, maxjoke-1, len(chars)), dtype=np.bool)
        for i in range(batch_size):
            start_index = random.randint(0, len(jokes)-1)
            sentence = jokes[start_index]
            jokelen = len(sentence)
            for j in range(maxjoke-1):
                x[i, j, char_indices[sentence[j%(jokelen-1)]]] = 1
                #x[i, j] = sentence[j%(jokelen-1)]
                y[i, j, char_indices[sentence[j%(jokelen-1) + 1]]] = 1
                #y[i, j] = sentence[j%(jokelen-1) + 1]

        yield (x, y)

    
generator = batch_generator(batch_size = batch_size)
####################################################################
#####################################################################
######################################################################
# Function for sampling characters on the output based on 
# their probabilites

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
###################################################################
###################################################################
###################################################################
# Callback function for saving the model and outputing its current
# work

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    savename = "models/jokegen_"+str(epoch)+".keras"
    print("Saving model...")
    model.save(savename)
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(jokes)-1)
    #start_index = random.randint(0, len(text) - maxlen - 1)
    maxxlen = len(jokes[start_index])-1
    newlen = random.randint(0, int(maxxlen/2))
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = jokes[start_index][0: newlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(MAX_JOKE_LENGTH):
            x_pred = np.zeros((1, len(sentence), len(chars))) # 1, newlen
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.   # 0, t, 

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds[len(preds)-1], diversity)
            next_char = indices_char[next_index]
            

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if next_char == "\n":
                break
        print()
###############################################################
###############################################################
###############################################################
###############################################################
# Build the corpus of all the characters preset in the data set

text = open('jokes_new.txt', 'r').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
################################################################
################################################################
################################################################
################################################################
# Make the list of all the jokes present in the data set and 
# extract the minimum and maximum length of the jokes present

filereader = open("jokes_new.txt", "r")
jokes = []
while True:
    newline = filereader.readline()
    if newline == "":
        break
    jokes.append(newline)
    
print("Total number of jokes in the dataset:", len(jokes))
maxjoke = 0
minjoke = 1000
for joke in jokes:
    newlen = len(joke)
    if newlen < minjoke:
        minjoke = newlen
    elif newlen > maxjoke:
        maxjoke = newlen
        
print("Length of the longest joke:", maxjoke)
print("Length of the shortest joke", minjoke)
#################################################################
##################################################################
###################################################################
####################################################################
# Building the model

print('Build the model...')
model = Sequential()
model.add(LSTM(units=NUM_OF_CELLS, return_sequences = True, input_shape=(None, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()
####################################################################

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Running the model
model.fit_generator(generator = generator,
                    epochs=NUM_OF_EPOCHS,
                    steps_per_epoch=NUM_OF_STEPS_PER_EPOCH,
                    callbacks=[print_callback])
