import keras
import random
import sys
from time import sleep
import numpy as np
from keras import layers

path = './data.txt'
text = open(path).read()


maxlen = 80
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])



chars = sorted(list(set(text)))

char_indices = dict((char, chars.index(char)) for char in chars)



def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model = keras.models.load_model('CixinLiu.h5')

length = int(input('Please enter how many chars you want to generate'))

   
start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index: start_index + maxlen]
print('\n--- Generating with seed: "' + generated_text + '"\n')

sys.stdout.write(generated_text)

for i in range(length):
    sampled = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(generated_text):
        sampled[0, t, char_indices[char]] = 1.

    preds = model.predict(sampled, verbose=0)[0]
    next_index = sample(preds, 0.8)
    next_char = chars[next_index]

    generated_text += next_char
    generated_text = generated_text[1:]

    sys.stdout.write(next_char)
    sys.stdout.flush()
    sleep(0.02)

sys.stdout.write('\n')
        
