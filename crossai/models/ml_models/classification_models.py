from ml_models.gru import TextGRU
from ml_models.lstm import TextBiLSTM
from ml_models.single_cnn import TextSingleChannelCNN
from ml_models.multi_cnn import TextMultichannelCNN
from ml_models.cnn_lstm import TextCNN_LSTM
from ml_models.cnn_gru import TextCNN_GRU


class Build_Classification_Model():
	def __init__(self, model_name, para):
		self.para = para
		self.model = None

		if model_name == 'gru':
			self.model = TextGRU(para)

		if model_name == 'lstm':
			self.model = TextBiLSTM(para)

		if model_name == 'cnn-gru':
			self.model = TextCNN_GRU(para)

		if model_name == 'cnn-lstm':
			self.model = TextCNN_LSTM(para)

		if model_name == 'single_cnn':
			self.model = TextSingleChannelCNN(para)

		if model_name == 'multi_cnn':
			self.model = TextMultichannelCNN(para)

	def fit(self):
		self.model.load_data()
		self.model.train()

	def predict(self, X):
		return self.model.predict(X)

	def predict_proba(self, X):
		return self.model.predict_proba(X)