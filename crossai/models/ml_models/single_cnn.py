import re
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D, Bidirectional, Flatten
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D, Normalization, Concatenate, MaxPool2D, Reshape, Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ml_models.base_model import BaseModel

class TextSingleChannelCNN(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        inp = tf.keras.Input(shape=(self.max_sequence_length,), dtype="int32")
        model = Sequential()
        x = Embedding(self.num_words,
                      self.embedding_dim,
                      input_length=self.max_sequence_length,
                      weights=[self.embedding_matrix],
                      trainable=self.trainable)(inp)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.4)(x)
        x = Dense(40, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=output)
        self.model = model