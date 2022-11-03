import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Flatten
from keras.layers import Dropout, GlobalMaxPool1D, GlobalAvgPool1D, Concatenate, MaxPool2D, Reshape, Conv2D
from ml_models.base_model import BaseModel


class TextMultichannelCNN(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.filter_sizes = [2, 3, 5]
        self.num_filters = 256

        inp = tf.keras.Input(shape=(self.max_sequence_length,), dtype='int32')
        model = Sequential()
        embedding = Embedding(self.num_words,
                        self.embedding_dim,
                        input_length=self.max_sequence_length,
                        weights=[self.embedding_matrix],
                        trainable=self.trainable)(inp)
        reshape = Reshape((self.max_sequence_length, self.embedding_dim, 1))(embedding)
        conv_0 = Conv2D(self.num_filters, 
                        kernel_size=(self.filter_sizes[0], self.embedding_dim), 
                        padding='valid', kernel_initializer='normal', 
                        activation='relu')(reshape)

        conv_1 = Conv2D(self.num_filters, 
                        kernel_size=(self.filter_sizes[1], self.embedding_dim), 
                        padding='valid', kernel_initializer='normal', 
                        activation='relu')(reshape)
        conv_2 = Conv2D(self.num_filters, 
                        kernel_size=(self.filter_sizes[2], self.embedding_dim), 
                        padding='valid', kernel_initializer='normal', 
                        activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(self.max_sequence_length - self.filter_sizes[0] + 1, 1), 
                            strides=(1,1), padding='valid')(conv_0)

        maxpool_1 = MaxPool2D(pool_size=(self.max_sequence_length - self.filter_sizes[1] + 1, 1), 
                            strides=(1,1), padding='valid')(conv_1)

        maxpool_2 = MaxPool2D(pool_size=(self.max_sequence_length - self.filter_sizes[2] + 1, 1), 
                            strides=(1,1), padding='valid')(conv_2)
        concatenated_tensor = Concatenate(axis=1)(
            [maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(0.3)(flatten)
        output = Dense(1, activation='sigmoid')(dropout)
        model = Model(inputs=inp, outputs=output)
        self.model = model