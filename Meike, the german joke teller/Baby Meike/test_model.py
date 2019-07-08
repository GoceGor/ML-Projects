from __future__ import print_function
from gtts import gTTS
from tkinter import *
import sys
import numpy as np
from keras.models import Sequential, load_model

#######################################################
# Configuration:
NUM_OF_JOKES = 5
MODEL_NUMBER = 49
MAX_JOKE_LENGTH = 250 
#######################################################

# Making the corpus:
text = open('jokes_new.txt', 'r').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


diversity = 0.5
chkbtns = []
all_jokes = []


# Helper functions
def sample(preds, temperature=1.0):
   preds = np.asarray(preds).astype('float64')
   preds = np.log(preds) / temperature
   exp_preds = np.exp(preds)
   preds = exp_preds / np.sum(exp_preds)
   probas = np.random.multinomial(1, preds, 1)
   return np.argmax(probas)

def fetch():
   global statusvar
   statusvar.set("Generating amazing jokes...")
   root.update_idletasks()
   jokestart = entry_joke.get()
   if len(jokestart) == 0:
      statusvar.set("Meike cannot finish an empty start of a joke... Try again!")
      root.update_idletasks()
      return 0
   if jokestart[len(jokestart) - 1] != " ":
      jokestart += " "

   global chkbtns
   global all_jokes
   all_jokes = []

   for mod in [MODEL_NUMBER]:
      modelpath = "models/jokegen_" + str(mod) + ".keras"
      model = load_model(modelpath)
      for kk in range(NUM_OF_JOKES):
         wholejoke = jokestart
         diversity = slider.get()

         generated = ''
         sentence = jokestart
         generated += sentence
         #sys.stdout.write(generated)

         for i in range(MAX_JOKE_LENGTH - len(jokestart)):
            x_pred = np.zeros((1, len(sentence), len(chars)))  
            for t, char in enumerate(sentence):
               x_pred[0, t, char_indices[char]] = 1. 

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds[len(preds) - 1], diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char
            wholejoke += next_char

            if next_char == "\n":
               break
         jokess[kk].set(wholejoke)
         all_jokes.append(wholejoke)

      del model
   statusvar.set("Amazing jokes generated, choose which one to play!")


def play():
   from pygame import mixer
   global statusvar
   statusvar.set("Playing the amazing joke...")
   root.update_idletasks()
   buttonstate = 0
   for kjh in range(len(chkbtns)):
      if chkbtns[kjh].get():
         buttonstate = kjh
         break
   language = 'de'
   myob = gTTS(text=all_jokes[buttonstate], lang=language, slow=False)
   myob.save('Joke.mp3')
   mixer.init()
   mixer.music.load("Joke.mp3")
   mixer.music.play()
   statusvar.set("How do you like the amazing joke? :D")
   root.update_idletasks()

#######################################################################
#           MAKING THE GUI:
#######################################################################

root=Tk()
root.geometry('1280x640')
root.title("Meike, the joke telling german : Baby phase")

main_container = Frame(root, background="bisque")
main_container.pack(side="top", fill="both", expand=True)

top_frame = Frame(main_container, background="green")
bottom_frame = Frame(main_container, background="yellow")
central_frame = Frame(main_container, background="red")
top_frame.pack(side="top", fill="both", expand=True)
central_frame.pack(side="top", fill="both", expand=True)
bottom_frame.pack(side="bottom", fill="x", expand=False)
        

top_left = Frame(top_frame, background="pink")
top_right = Frame(top_frame, background="blue")
top_left.pack(side="left", fill="both", expand=True)
top_right.pack(side="right", fill="both", expand=True)


label1 = Label(top_left, text = "Enter the start of the joke!")
label1.config(font=("Courier", 20))
entry_joke = Entry(top_left)

top_right_label = Label(top_right, text="Top Right")
label_slider = Label(top_right, text = "Enter the level of creativity you desire!")
label_slider.config(font=("Courier", 20))
slider = Scale(top_right, orient = 'horizontal', from_ = 0.05, to = 1.5, resolution = 0.05)

button_generate = Button(top_left, text = "Generate!",bg='brown',fg='white', command=fetch)
button_generate.config(font=("Courier", 20))
button_play = Button(top_left, text = "Play ze joke!",bg='brown',fg='white', command=play)
button_play.config(font=("Courier", 20))

statusvar = StringVar()
status = Label(bottom_frame, textvariable = statusvar, bg='brown',fg='white')
status.config(font=("Courier", 16))
statusvar.set("Ready to generate amazing jokes!")


label1.pack(side="top", fill = "both", expand = True)
entry_joke.pack(side="top", fill = "both", expand = True)
button_play.pack(side = "bottom", fill = "both", expand = True)
button_generate.pack(side = "bottom", fill = "both", expand = True)

label_slider.pack(side="top", fill="both", expand=True)
slider.pack(side="top", fill="both", expand=True)

status.pack(side="left", fill="both", expand=True)

jokess = []
for k in range(NUM_OF_JOKES):
   jokess.append(StringVar())

for k in range(NUM_OF_JOKES):
   chkbtns.append(BooleanVar())
   Checkbutton(central_frame, variable = chkbtns[k], textvariable = jokess[k], bg='white',fg='brown', anchor = W).pack(side = "top", fill = "both", expand = True)



root.mainloop()
