import re
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D, Bidirectional, Flatten, concatenate
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D, Normalization, Concatenate, MaxPool2D, Reshape, Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ml_models.base_model import BaseModel

class TextCNN_GRU(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        inp = tf.keras.Input(shape=(self.max_sequence_length,), dtype="int32")
        model = Sequential()
        x = Embedding(self.num_words,
                      self.embedding_dim,
                      input_length=self.max_sequence_length,
                      weights=[self.embedding_matrix],
                      trainable=self.trainable)(inp)
        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(GRU(100, return_sequences=True))(x)
        x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
        avg_pool = GlobalAvgPool1D()(x)
        max_pool = GlobalMaxPool1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(1, activation="sigmoid")(conc)
        model = Model(inputs=inp, outputs=outp)
        self.model = model