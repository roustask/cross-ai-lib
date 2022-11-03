from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten, GRU, concatenate
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D, Normalization, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, config, **kwargs):
        """Initialize model parameters

        Args:
            config (dict): model configuration
        """
        self.config = config
        self.batch_size = self.config['model']['batch_size']
        self.epochs = self.config['model']['epochs']
        self.max_sequence_length = self.config['model']['max_sequence_length']
        self.num_words = self.config['model']['num_words']
        self.embedding_dim = self.config['model']['embedding_dim']
        self.embedding_matrix = self.config['model']['embedding_matrix']
        self.trainable = self.config['model']['trainable']
        self.callbacks = []
        if self.config['model']['early_stopping']:
            self.early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            self.callbacks.append(self.early_stopping)
        if self.config['model']['reduce_lr']:
            self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=2, min_lr=0.001)
            self.callbacks.append(self.reduce_lr)
        self.checkpoint_filepath = './tmp/checkpoint'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        self.callbacks.append(self.model_checkpoint_callback)
        self.data = self.config['data']
        self.model = None

    def load_data(self):
        """Load data from config['data']"""
        self.X_train = self.data['X_train']
        self.y_train = self.data['Y_train']
        self.X_test = self.data['X_eval']
        self.y_test = self.data['Y_eval']

    def train(self):
        """Train model

        Returns:
            history: model training history
        """
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.callbacks])
        return history

    def predict(self, X):
        """Predict on X

        Args:
            X : input data

        Returns:
            y_pred: predicted labels
        """        
        self.model.load_weights(self.checkpoint_filepath)
        y_pred = self.model.predict(X)
        y_pred = np.round(y_pred).astype(int)
        return y_pred

    def predict_proba(self, X):
        """Predict probabilities on X

        Args:
            X : input data

        Returns:
            y_pred: predicted probabilities
        """        
        self.model.load_weights(self.checkpoint_filepath)
        y_pred = self.model.predict(X)
        return y_pred