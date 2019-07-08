# Baby Meike


## 1 Introduction

The 'light' version of Meike's sense of humor is her Baby version that only uses a Character-Based version of a Recurrent Neural Network (LSTM). Given that she was a baby at that age, Meike was only able to understand letters and dependencies between them, enough to be able to make words in english, but not quite enough to capture the essence what a joke is. Her parents told her that her jokes make sense, but some people claim that their parents were just too sensitive. What majority of them agree with is that she definitely showed glimpses of her joke-telling potential even at such a young age. Give her the start of a joke, let her finish it and tell you the joke, and decide for yourself. 


## 2 More "technical" explanation

In case you are only interested in trying the project, without wishing to find out more about how it functions, feel free to skip to point 3.

This version of the project is intended to work as a "beta" version. The process of training was done using the *quick-and-dirty* approach, where the performance of the model is not of utmost importance, but getting a quick solution to be used as proof of concept. A more advanced approach that should yield better results will be introduced in a project representing an "older" version of Meike.

The code for training the model is a changed version of the code that was given as example for LSTM text generation on the Keras website at the next address:

https://keras.io/examples/lstm_text_generation/ 

, while the data used is a text file of about 200k jokes, that is basically a slightly changed version of the Short jokes dataset that can be found on this address:

https://github.com/amoudgl/funnybot


The project is divided into 2 main scripts, one for training the model and one for testing its performance:

### 2.1 Training script

The training script is called train_model.py. It first starts with the configuration, where values are given to the constants that are used throughout the script. Changing the values of these constants should have effect on the performance of the model in the end. Rest of the script is a changed version of the aforementioned example script on the Keras website.

The training script used in this project trains a character-based LSTM model, that is only able of outputing one letter at a time, and as such does not learn more complex features of a joke, but at best learns to reproduce the words that it got to *see* in the training set the most.

### 2.2 Test script

The test script was written using Tkinter, with the purpose of making the results of training more representable. Its functionality is the same as the functionality of the part of the code that would be executed on epoch end during the training process, except that, instead of using a the beginning of a joke as the *seed* for the joke creation, it uses the input that the user has given. 

The project already comes pretrained with 50 models saved, one after each epoch. In general, the higher the epoch number, the better the model should be at creating jokes, but, the model converges pretty fast, so all the models after the 10th-15th epoch have more-less similar performance. Feel free to play with the number of the model, number of jokes, and maximum length (in characters) that the model should produce.

## 3 Instructions

First step is to check whether all the dependencies are installed:

1. Tensorflow and Keras (versions used are 1.5 for Tensorflow and 2.2.4 for Keras)

2. gTTS

3. Tkinter

4. Numpy

If that is the case, run the *test_model.py* script. A window will open. In the upper-left part of the screen, you can input the start of the joke. One more parameter worth noting is the creativity parameter, In general, the higher the creativity, the more likely the model is to make creative jokes, but also, more likely to come up with words that have no meaning in english (and very likely in any other language). 

Once the input and creativity are set, press the "Generate" button.

After some time, the model will generate and display 5 jokes, next to their respective checkbuttons. Choose the joke that you like the most, press the "Play" button, and enjoy.

*Note* - In case of several jokes being checked, only the one closest to the top of the screen will be played.
