import numpy as np
from tensorflow import keras
import logging
from crossai.models.nn1d.base_model import BaseModel


class BiLSTM(BaseModel):
    def define_model(self, num_of_classes=1, lstm_units=None,
                     n_layers=None, input_shape=None, dropout_percent=0,
                     dense_layers=None, dense_units=None):
        self.arch.num_of_classes = num_of_classes
        self.arch.lstm_units = lstm_units
        self.arch.n_layers = n_layers
        self.arch.input_shape = input_shape
        self.arch.dropout_percent = dropout_percent
        self.arch.dense_layers = dense_layers
        self.arch.dense_units = dense_units


        debug_msg_str = "model name : " + self.model_name + "\n"
        debug_msg_str = debug_msg_str + "shaping : "+repr((None, self.arch.input_shape[0],
                                                           self.arch.input_shape[1])) + "\n"
        debug_msg_str = debug_msg_str + "n_layers : " + repr(self.arch.n_layers) + "\n"
        debug_msg_str = debug_msg_str + "LSTM units : " + repr(self.arch.lstm_units) + "\n"
        debug_msg_str = debug_msg_str + "Dense units : " + repr(self.arch.dense_units) + "\n"
        logging.debug(debug_msg_str)
        self.model = keras.models.Sequential(name=self.model_name)
        self.model.add(keras.layers.Input(shape=self.arch.input_shape))
        self.model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(units=self.arch.lstm_units[0], activation="tanh",
                              return_sequences=True)))

        self.model.add(keras.layers.Dropout(self.arch.dropout_percent))
        for layer in range(0,self.arch.n_layers-1):
            self.model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(units=self.arch.lstm_units[layer],
                                  return_sequences=True, activation="tanh")))
            if layer == 0:
                self.model.add(keras.layers.Dropout(self.arch.dropout_percent))
        self.model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(units=self.arch.lstm_units[layer],
                              return_sequences=False, activation="tanh")))
        for d in range(0, self.arch.dense_layers):
            self.model.add(keras.layers.Dense(units=self.arch.dense_units[d],
                                              activation="relu"))
        self.model.add(keras.layers.Dense(self.arch.num_of_classes,
                                          activation="softmax"))

    def predictions_on_testset(self, test_X, verbose=0):
        """

        Args:
            test_X:
            verbose:

        Returns:

        """
        logging.info("Model (BiLSTM) prediction on testset.")
        predictions = self.model.predict(
            test_X,
            verbose=1 if logging.root.level == logging.DEBUG else 0)
        # return np.argmax(np.sum(predictions, axis=1), axis=1)
        return np.argmax(predictions, axis=1)
