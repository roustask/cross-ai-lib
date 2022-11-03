import re
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D, Bidirectional, Flatten
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D, Normalization, Concatenate, MaxPool2D, Reshape, Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ml_models.base_model import BaseModel

class TextCNN_LSTM(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        inp = tf.keras.Input(shape=(self.max_sequence_length,), dtype="int32")
        model = Sequential()
        x = Embedding(self.num_words,
                      self.embedding_dim,
                      input_length=self.max_sequence_length,
                      weights=[self.embedding_matrix],
                      trainable=self.trainable)(inp)
        x = Bidirectional(LSTM(100, dropout=0.3, return_sequences=True))(x)
        x = Bidirectional(LSTM(100, dropout=0.3, return_sequences=True))(x)
        x = Conv1D(100, 5, activation='relu')(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(16, activation='relu')(x)
        outp = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=outp)
        self.model = model