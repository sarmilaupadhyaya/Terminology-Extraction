import numpy as np
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import math

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sen_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.floor(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class Dataloader:

    def __init__(self, dframe,tag2idx, maxlen, batchsize, ntags):

        getter = SentenceGetter(dframe)
        self.sentences = getter.sentences
        self.tag2idx = tag2idx
        self.batch_size = batchsize
        self.maxlen = maxlen
        self.n_tags=ntags
        



    def loader(self):
        X = [[w[0].lower() for w in s] for s in self.sentences]
        new_X = []
        for seq in X:
            new_seq = []
            for i in range(self.maxlen):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("__PAD__")
            new_X.append(new_seq)
        X = new_X

        y_idx = [[self.tag2idx[w[1]] for w in s] for s in self.sentences]
        y = pad_sequences(maxlen=self.maxlen, sequences=y_idx, padding="post", value=self.tag2idx["O"])
        y = [to_categorical(i, num_classes=self.n_tags) for i in y]
        self.y = y

        return Generator(np.array(X), np.array(y), self.batch_size)















