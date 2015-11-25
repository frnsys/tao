import os
import json
import random
import numpy as np
from glob import glob
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout


class Tao():
    def __init__(self):
        texts = []
        for f in glob('docs/output/**/*.txt'):
            texts.append(open(f, 'r').read())
        self.text = '\n'.join(texts)
        self.max_len = 20

    def save(self):
        print('Saving model...')
        self.model.save_weights('model/weights.h5', overwrite=True)

        with open('model/architecture.json', 'w') as f:
            f.write(self.model.to_json())

        meta = {
            'chars': self.chars,
            'char_idxs': self.char_idxs,
            'idxs_char': self.idxs_char,
            'max_len': self.max_len
        }
        with open('model/meta.json', 'w') as f:
            json.dump(meta, f)

    def load(self):
        print('Loading model...')
        with open('model/meta.json', 'r') as f:
            meta = json.load(f)

        self.chars = meta['chars']
        self.char_idxs = meta['char_idxs']
        self.idxs_char = meta['idxs_char']
        self.max_len = meta['max_len']

        with open('model/architecture.json', 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights('model/weights.h5')

    def _build_model(self):
        print('Building model...')

        self.chars = list(set(self.text))
        print('\tn chars', len(self.chars))

        self.char_idxs = {}
        self.idxs_char = {}
        for i, c in enumerate(self.chars):
            self.char_idxs[c] = i
            self.idxs_char[i] = c

        self.model = Sequential()
        self.model.add(LSTM(512, return_sequences=True, input_shape=(self.max_len, len(self.chars))))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(512, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def generate(self, temperature=0.35, seed=None, predicate=lambda x: len(x) < 400):
        if seed is None:
            start_idx = random.randint(0, len(self.text) - self.max_len - 1)
            seed = self.text[start_idx:start_idx + self.max_len]

        sentence = seed
        generated = sentence

        while predicate(generated):
            x = np.zeros((1, self.max_len, len(self.chars)))
            for t, char in enumerate(sentence):
                x[0, t, self.char_idxs[char]] = 1.

            preds = self.model.predict(x, verbose=0)[0]
            next_idx = self._sample(preds, temperature)
            next_char = self.idxs_char[next_idx]

            generated += next_char
            sentence = sentence[1:] + next_char
        return generated

    def train(self, epochs=10, save=True):
        if os.path.exists('model/meta.json'):
            self.load()
        else:
            self._build_model()

        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.max_len, step):
            sentences.append(self.text[i:i + self.max_len])
            next_chars.append(self.text[i + self.max_len])
        print('\tn seqs', len(sentences))

        X = np.zeros((len(sentences), self.max_len, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sent in enumerate(sentences):
            for t, char in enumerate(sent):
                X[i, t, self.char_idxs[char]] = 1
            y[i, self.char_idxs[next_chars[i]]] = 1

        print('Training model...')

        for i in range(epochs):
            print('Epoch', i)
            self.model.fit(X, y, batch_size=128, nb_epoch=1)

            # preview
            for temp in [0.2, 0.5, 1., 1.2]:
                print('\n\ttemperature:', temp)
                self.generate(temperature=temp)

        if save:
            print('Saving model...')
            self.save()

    def _sample(self, a, temperature=1.0):
        """sample an index from a probability array"""
        a = np.log(a)/temperature
        a = np.exp(a)/np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    def the_way(self):
        n_parts = 81
        seed = 'The Tao is the way.\n'
        return self.generate(seed=seed,
                             predicate=lambda x: x.split('\n').count('') <= n_parts)
