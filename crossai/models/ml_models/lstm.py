import re
import tensorflow as tf
from ml_models.base_model import BaseModel

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten, GRU, concatenate
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D, Normalization, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ml_models.base_model import BaseModel


class TextBiLSTM(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        inp = tf.keras.Input(shape=(self.max_sequence_length,), dtype="int32")
        model = Sequential()
        x = Embedding(self.num_words,
                      self.embedding_dim,
                      input_length=self.max_sequence_length,
                      weights=[self.embedding_matrix],
                      trainable=self.trainable)(inp)
        x = Bidirectional(LSTM(64,  return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dense(150, activation='relu')(x)
        x = Dropout(0.4)(x)
        output = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=output)
        self.model = model
