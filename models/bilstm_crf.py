## code to train the bilstm crf model


import params
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
import keras as k
import tensorflow as tf
import tensorflow_hub as hub
from keras_contrib.layers import CRF
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


class ElmoBilstmCrf:

    def __init__(self, max_len, batch_size, word_embedding_size, n_tags=2):
        
        self.max_len = max_len
        self.batch_size = batch_size
        self.word_embedding_size = word_embedding_size
        self.n_tags = n_tags
        self.model = self.get_model()

    def ElmoEmbedding(self,x):

        return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, 'string')),
                            "sequence_len": tf.constant(self.batch_size*[self.max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]

    def get_model(self):

        input = Input(shape=(self.max_len,),dtype='string')
        model = Lambda(self.ElmoEmbedding, output_shape=(self.max_len, 1024))(input)

        model = Bidirectional(LSTM(units=1024,
                           return_sequences=True,
                           dropout=0.5,
                           recurrent_dropout=0.5,
                           kernel_initializer=k.initializers.he_normal()))(model)
        model = LSTM(units=1024 * 2,
             return_sequences=True,
             dropout=0.5,
             recurrent_dropout=0.5,
             kernel_initializer=k.initializers.he_normal())(model)
        model = TimeDistributed(Dense(self.n_tags, activation="relu"))(model)  # previously softmax output layer
        crf = CRF(self.n_tags)  # CRF layer
        out = crf(model)  # output
        model = Model(input, out)
        adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])

        return model



