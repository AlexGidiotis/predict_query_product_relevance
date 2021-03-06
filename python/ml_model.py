from pyspark.sql import SparkSession
import numpy as np
import json

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Dropout, Convolution1D
from keras.layers import Embedding, Flatten, Input
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras import regularizers
from keras import backend as K

from sklearn.model_selection import train_test_split


embedding_dim = 200
glove_path = '/Users/alexgidiotis/atypon/explorer-ai/glove.6B/glove.6B.200d.txt'
STAMP = 'relevance_predictor'


def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


def load_wrd_embeddings(glove_path,
	embedding_dim):

	json_file = open('data/word_index.json')
	json_string = json_file.read()
	wrd2id = json.loads(json_string)
	vocab_size = len(wrd2id)
	print "Found %s words in the vocabulary." % vocab_size

	embedding_idx = {}
	glove_f = open(glove_path)
	for line in glove_f:
		values = line.split()
		wrd = values[0]
		coefs = np.asarray(values[1:],
			dtype='float32')
		embedding_idx[wrd] = coefs
	glove_f.close()
	print "Found %s word vectors." % len(embedding_idx)

	embedding_mat = np.zeros((vocab_size+1,embedding_dim))
	for wrd, i in wrd2id.items():
		embedding_vec = embedding_idx.get(wrd)
		# words without embeddings will be left with zeros.
		if embedding_vec is not None:
			embedding_mat[i] = embedding_vec

	print embedding_mat.shape
	return embedding_mat, vocab_size


def build_net(title_len,
	sterm_len,
	embedding_dims,
	lr=0.001):
	
	embedding_mat,vocab_size = load_wrd_embeddings(glove_path,embedding_dims)

	# Title block
	input_layer_1 = Input(shape=(title_len,),
		dtype='int32')

	title_embedding_layer = Embedding(input_dim=vocab_size+1,
		output_dim=embedding_dims,
		input_length=title_len,
		weights=[embedding_mat],
		trainable=False)(input_layer_1)

	title_drop1 = Dropout(0.3)(title_embedding_layer)

	title_conv1 = Convolution1D(32, (2),
		activation='relu',
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3),
		kernel_regularizer=regularizers.l2(0.0))(title_drop1)
	title_conv1 = MaxPooling1D(2)(title_conv1)

	title_conv2 = Convolution1D(32, (4),
		activation='relu',
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3),
		kernel_regularizer=regularizers.l2(0.0))(title_drop1)
	title_conv2 = MaxPooling1D(2)(title_conv2)

	title_conv3 = Convolution1D(32, (8),
		activation='relu',
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3),
		kernel_regularizer=regularizers.l2(0.0))(title_drop1)
	title_conv3 = MaxPooling1D(2)(title_conv3)
	
	title_concat = concatenate([title_conv1, title_conv2, title_conv3], axis=1)

	title_flat = Flatten()(title_concat)


	# Sterm block
	input_layer_2 = Input(shape=(sterm_len,),
		dtype='int32')

	sterm_embedding_layer = Embedding(input_dim=vocab_size+1,
		output_dim=embedding_dims,
		input_length=sterm_len,
		weights=[embedding_mat],
		trainable=False)(input_layer_2)

	sterm_drop1 = Dropout(0.3)(sterm_embedding_layer)

	sterm_conv1 = Convolution1D(32, (2),
		activation='relu',
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3),
		kernel_regularizer=regularizers.l2(0.0))(sterm_drop1)
	sterm_conv1 = MaxPooling1D(2)(sterm_conv1)

	sterm_conv2 = Convolution1D(32, (4),
		activation='relu',
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3),
		kernel_regularizer=regularizers.l2(0.0))(sterm_drop1)
	sterm_conv2 = MaxPooling1D(2)(sterm_conv2)

	sterm_conv3 = Convolution1D(32, (8),
		activation='relu',
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3),
		kernel_regularizer=regularizers.l2(0.0))(sterm_drop1)
	sterm_conv3 = MaxPooling1D(2)(sterm_conv3)
	
	sterm_concat = concatenate([sterm_conv1, sterm_conv2, sterm_conv3], axis=1)

	sterm_flat = Flatten()(sterm_concat)

	# Combined block
	concat = concatenate([title_flat,sterm_flat], axis=1)

	dense1 = Dense(512,
		activation='relu',
		kernel_initializer='lecun_uniform')(concat)

	drop2 = Dropout(0.5)(dense1)

	predictions = Dense(1, activation='relu')(drop2)

	model = Model(inputs=[input_layer_1, input_layer_2], outputs=predictions)

	adam = Adam(lr=lr,
		decay=1e-5)

	model.compile(loss=rmse,
		optimizer=adam,
		metrics=[])

	model.summary()

	return model


def train_model():
	max_title_len = 15
	max_sterm_len = 10

	spark = SparkSession.builder.getOrCreate()

	train_df = spark.read.json('data/train_set')
	train_labs_df = spark.read.json('data/train_set_labels')

	X_titles = [item.int_title_tokens for item in train_df.select('int_title_tokens').collect()]
	X_sterms = [item.int_sterm_tokens for item in train_df.select('int_sterm_tokens').collect()]
	y_data = [float(item.relevance) for item in train_labs_df.select('relevance').collect()]

	X_titles_train,X_titles_val,y_train,y_val = train_test_split(X_titles,y_data,
		test_size=0.2,
		random_state=10)

	X_sterms_train,X_sterms_val,y_train,y_val = train_test_split(X_sterms,y_data,
		test_size=0.2,
		random_state=10)

	spark.stop()

	print('Average train title sequence length: {}'.format(np.mean(list(map(len, X_titles_train)), dtype=int)))
	print('Average train sterm sequence length: {}'.format(np.mean(list(map(len, X_sterms_train)), dtype=int)))

	X_titles_train = pad_sequences(X_titles_train,
		maxlen=max_title_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_sterms_train = pad_sequences(X_sterms_train,
		maxlen=max_sterm_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_titles_val = pad_sequences(X_titles_val,
		maxlen=max_title_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_sterms_val = pad_sequences(X_sterms_val,
		maxlen=max_sterm_len,
		padding='post',
		truncating='post',
		dtype='float32')

	print X_titles_train.shape, X_sterms_train.shape

	model = build_net(max_title_len,max_sterm_len,embedding_dim)

	model_json = model.to_json()
	with open("model/" + STAMP + ".json", "w") as json_file:
		json_file.write(model_json)

	early_stopping = EarlyStopping(monitor='val_loss',
		patience=5)
	bst_model_path = "model/" + STAMP + '.h5'
	model_checkpoint = ModelCheckpoint(bst_model_path,
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=True)

	hist = model.fit([X_titles_train,X_sterms_train],y_train,
		validation_data=([X_titles_val,X_sterms_val],y_val),
		epochs=10,
		batch_size=64,
		shuffle=True,
		callbacks=[early_stopping,model_checkpoint],
		verbose=1)

	model.load_weights('model/' + STAMP + '.h5')
	predictions = model.predict([X_titles_val,X_sterms_val])

	print predictions[:20]
	print y_val[:20]


def load_model(STAMP=STAMP):
	"""
	Loads the trained model and weights.

	Arguments:
		STAMP: The STAMP of the model.

	Returns:
		loaded_model:
	"""
	# Load the model architecture.
	json_file = open('model/' + STAMP + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('model/' + STAMP + '.h5')
	print("Loaded model from disk")
	return loaded_model


def evaluate_model():
	max_title_len = 15
	max_sterm_len = 10


	spark = SparkSession.builder.getOrCreate()

	train_df = spark.read.json('data/train_set')

	X_titles = [item.int_title_tokens for item in train_df.select('int_title_tokens').collect()]
	X_sterms = [item.int_sterm_tokens for item in train_df.select('int_sterm_tokens').collect()]

	train_labs_df = spark.read.json('data/train_set_labels')
	y_data = [float(item.relevance) for item in train_labs_df.select('relevance').collect()]

	spark.stop()

	X_titles_train,X_titles_val,y_train,y_val = train_test_split(X_titles,y_data,
		test_size=0.2,
		random_state=10)

	X_sterms_train,X_sterms_val,y_train,y_val = train_test_split(X_sterms,y_data,
		test_size=0.2,
		random_state=10)

	print('Average train title sequence length: {}'.format(np.mean(list(map(len, X_titles_train)), dtype=int)))
	print('Average train sterm sequence length: {}'.format(np.mean(list(map(len, X_sterms_train)), dtype=int)))

	X_titles_train = pad_sequences(X_titles_train,
		maxlen=max_title_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_sterms_train = pad_sequences(X_sterms_train,
		maxlen=max_sterm_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_titles_val = pad_sequences(X_titles_val,
		maxlen=max_title_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_sterms_val = pad_sequences(X_sterms_val,
		maxlen=max_sterm_len,
		padding='post',
		truncating='post',
		dtype='float32')

	print X_titles_train.shape, X_sterms_train.shape

	model = load_model('relevance_predictor')

	predictions = model.predict([X_titles_val,X_sterms_val])



	print 'writing output'
	with open("model/evaluation_cnn.csv", "w") as csv_file:
		csv_file.write('label,prediction\n')
		for id_num,pred in zip(y_val,predictions):
			csv_file.write('%f ,%f \n' %(id_num,pred))


def predict_model():
	max_title_len = 15
	max_sterm_len = 10


	spark = SparkSession.builder.getOrCreate()

	train_df = spark.read.json('data/new_set')

	X_titles_train = [item.int_title_tokens for item in train_df.select('int_title_tokens').collect()]
	X_sterms_train = [item.int_sterm_tokens for item in train_df.select('int_sterm_tokens').collect()]

	train_labs_df = spark.read.json('data/new_set_ids')
	y_train = [float(item.id) for item in train_labs_df.select('id').collect()]

	spark.stop()

	print('Average train title sequence length: {}'.format(np.mean(list(map(len, X_titles_train)), dtype=int)))
	print('Average train sterm sequence length: {}'.format(np.mean(list(map(len, X_sterms_train)), dtype=int)))

	X_titles_train = pad_sequences(X_titles_train,
		maxlen=max_title_len,
		padding='post',
		truncating='post',
		dtype='float32')

	X_sterms_train = pad_sequences(X_sterms_train,
		maxlen=max_sterm_len,
		padding='post',
		truncating='post',
		dtype='float32')

	print X_titles_train.shape, X_sterms_train.shape

	model = load_model('relevance_predictor')

	predictions = model.predict([X_titles_train,X_sterms_train])

	print 'writing output'
	with open("model/predicted_results_cnn.csv", "w") as csv_file:
		csv_file.write('id,prediction\n')
		for id_num,pred in zip(y_train,predictions):
			csv_file.write('%d ,%f \n' %(id_num,pred))

if __name__ == '__main__':
	#predict_model()
	#train_model()
	evaluate_model()