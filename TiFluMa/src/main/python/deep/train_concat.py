
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from scipy import sparse
from scipy.sparse import csr_matrix

import keras_metrics as km
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Dense, Dropout, Embedding, Flatten,
                          Input, MaxPooling1D, concatenate)
from keras.models import Model, Sequential
from keras.optimizers import Adadelta
from keras.utils import np_utils, plot_model, to_categorical
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.python.client import device_lib

matplotlib.use('Agg')

K.tensorflow_backend._get_available_gpus()


class Metrics:

    epoch_writer_file_path: str = ""
    metric: dict = dict()

    def initialise_epoch_writer(self, model_name: str):
        self.epoch_writer_file_path = f'{model_name}_result.csv'
        self.write_epoch_header()

    def write_epoch_header(self):
        with open(self.epoch_writer_file_path, 'w') as my_file:
            my_file.write('Precision Recall Macro-f-score Micro-f-score')

    def write_epoch_metrics(self):
        with open(self.epoch_writer_file_path, 'a') as my_file:
            my_file.write(
                f"\n{self.metric['prec']} {self.metric['reca']} {self.metric['f1_mac']} {self.metric['f1_mic']}")

    def initialise_best_result_writer(self, model_name: str):
        with open(f'{model_name}_best_result.csv', 'w') as my_file:
            my_file.write('Precision Recall Macro-f-score Micro-f-score Type')

    def write_best_result(self, model_name: str, data_type: str = ''):
        with open(f'{model_name}_best_result.csv', 'a') as my_file:
            my_file.write(
                f"\n{self.metric['prec']} {self.metric['reca']} {self.metric['f1_mac']} {self.metric['f1_mic']} {data_type}")

    def calculate_metrics(self, predicted, gold):
        self.metric.clear()
        self.metric['prec'] = precision_score(gold, predicted, average='micro')
        self.metric['reca'] = recall_score(gold, predicted, average='micro')
        self.metric['accu'] = accuracy_score(gold, predicted)
        self.metric['f1_mac'] = f1_score(gold, predicted,  average='macro')
        self.metric['f1_mic'] = f1_score(gold, predicted,  average='micro')

    def print_result_scores(self):
        print("Precision: ", self.metric['prec'])
        print("Accuracy: ",  self.metric['accu'])
        print("Macro F1-score: ",  self.metric['f1_mac'])
        print("Micro F1-score: ",  self.metric['f1_mic'])


class Net:

    encoder = LabelBinarizer()

    # features
    stylometric_train_features: csr_matrix = None
    stylometric_eval_features: csr_matrix = None
    stylometric_test_features: csr_matrix = None
    stylometric_eval_features_path: str = ""
    stylometric_train_features_path: str = ""
    stylometric_test_features_path: str = ""

    rhyme_train_features: csr_matrix = None
    rhyme_eval_features: csr_matrix = None
    rhyme_test_features: csr_matrix = None
    rhyme_eval_features_path: str = ""
    rhyme_train_features_path: str = ""
    rhyme_test_features_path: str = ""

    word_count_train_features: csr_matrix = None
    word_count_eval_features: csr_matrix = None
    word_count_test_features: csr_matrix = None
    word_count_eval_features_path: str = ""
    word_count_train_features_path: str = ""
    word_count_test_features_path: str = ""

    pos_count_test_features: csr_matrix = None
    pos_count_train_features: csr_matrix = None
    pos_count_eval_features: csr_matrix = None
    pos_count_eval_features_path: str = ""
    pos_count_train_features_path: str = ""
    pos_count_test_features_path: str = ""

    word_embedd_test_features: csr_matrix = None
    word_embedd_train_features: csr_matrix = None
    word_embedd_eval_features: csr_matrix = None
    word_embedd_eval_features_path: str = ""
    word_embedd_train_features_path: str = ""
    word_embedd_test_features_path: str = ""

    # labels
    test_labels_path: str = ""
    train_labels_path: str = ""
    eval_labels_path: str = ""
    eval_labels = None
    train_labels = None
    test_labels = None

    # features to use
    use_stylometric = True
    use_rhyme = False
    use_word_count = False
    use_pos_count = True
    use_word_embedd = True

    # hyperparameters:
    vocab_size = 0
    learning_rate = 1.0
    rho = 0.95
    epochs = 200
    maxlen = 100
    embedding_dim = 50
    embedding_matrix = None
    batch_size = 64
    loss_function = 'categorical_crossentropy'

    model_name = 'myModel'
    model = None

    def load_test_data(self):
        """
        Loads the test data.
        """
        print("Loading data...")
        print("Reading stylometric data...")
        self.stylometric_test_features = sparse.load_npz(
            self.stylometric_test_features_path)

        print("Reading rhyme data...")
        self.rhyme_test_features = sparse.load_npz(
            self.rhyme_test_features_path)

        print("Reading word count data...")
        self.word_count_test_features = sparse.load_npz(
            self.word_count_test_features_path)

        print("Reading POS data...")
        self.pos_count_test_features = sparse.load_npz(
            self.pos_count_test_features_path)

        print("Reading word embedding data...")
        self.word_embedd_test_features = sparse.load_npz(
            self.word_embedd_test_features_path)

        print("Reading label data...")
        self.test_labels = pd.read_pickle(self.test_labels_path)

    def load_data(self):
        """
        Loads the training and evaluation data.
        """
        print("Loading data...")
        print("Reading stylometric data...")
        self.stylometric_train_features = sparse.load_npz(
            self.stylometric_train_features_path)
        self.stylometric_eval_features = sparse.load_npz(
            self.stylometric_eval_features_path)

        print("Reading rhyme data...")
        self.rhyme_train_features = sparse.load_npz(
            self.rhyme_train_features_path)
        print(self.rhyme_train_features)
        self.rhyme_eval_features = sparse.load_npz(
            self.rhyme_eval_features_path)

        print("Reading word count data...")
        self.word_count_train_features = sparse.load_npz(
            self.word_count_train_features_path)
        self.word_count_eval_features = sparse.load_npz(
            self.word_count_eval_features_path)

        print("Reading POS data...")
        self.pos_count_train_features = sparse.load_npz(
            self.pos_count_train_features_path)
        self.pos_count_eval_features = sparse.load_npz(
            self.pos_count_eval_features_path)

        print("Reading word embedding data...")
        self.word_embedd_train_features = sparse.load_npz(
            self.word_embedd_train_features_path)
        self.word_embedd_eval_features = sparse.load_npz(
            self.word_embedd_eval_features_path)

        print("Reading label data...")
        self.train_labels = pd.read_pickle(self.train_labels_path)
        self.eval_labels = pd.read_pickle(self.eval_labels_path)

    def encode_labels(self):
        """ Encodes multi-class traing and eval labels to binary labels
        """
        print("Encoding labels")
        self.train_labels = self.encoder.fit_transform(
            self.train_labels)

        self.eval_labels = self.encoder.fit_transform(
            self.eval_labels)

    def build_concatenated_model(self):
        """Builds a concatenated model
        """

        model_inputs = []
        to_concatenate = []
        if self.use_stylometric:
            stylometric_model = self.define_stylometric_model()
            to_concatenate.append(stylometric_model.output)
            model_inputs.append(stylometric_model.input)
        if self.use_rhyme:
            rhyme_model = self.define_rhyme_model()
            to_concatenate.append(rhyme_model.output)
            model_inputs.append(rhyme_model.input)
        if self.use_word_count:
            word_count_model = self.define_word_count_model()
            to_concatenate.append(word_count_model.output)
            model_inputs.append(word_count_model.input)
        if self.use_pos_count:
            pos_count_model = self.define_pos_count_model()
            to_concatenate.append(pos_count_model.output)
            model_inputs.append(pos_count_model.input)
        if self.use_word_embedd:
            word_embedd_model = self.define_word_embedd_model()
            to_concatenate.append(word_embedd_model.output)
            model_inputs.append(word_embedd_model.input)

        if len(to_concatenate) > 1:
            combined = concatenate(to_concatenate)
        else:
            combined = to_concatenate[0]

        d = Dropout(0.2, input_shape=(60,))(combined)
        z = Dense(128, activation="relu", name='dense_9')(d)
        z = Dropout(0.2, input_shape=(60,))(z)

        z = Dense(self.train_labels.shape[1],
                  activation="softmax", name='end_dense_1')(z)
        self.model = Model(inputs=model_inputs, outputs=z)

    def define_stylometric_model(self) -> Model:
        print(self.train_labels.shape[1])
        input_dim = self.stylometric_train_features.shape[1]
        inputs = Input(shape=(input_dim,), name='stylometric_input')
        x = Dense(16, activation='relu', name='stylometric_dense_0')(inputs)
        x = Dense(8, activation='relu', name='stylometric_dense_1')(x)
        model = Model(inputs, x)
        return model

    def define_rhyme_model(self) -> Model:
        print(self.rhyme_train_features.shape)
        input_dim = self.rhyme_train_features.shape[1]
        inputs = Input(shape=(input_dim,), name='rhyme_input')
        x = Dense(64, activation='relu', name='rhyme_dense_0')(inputs)
        model = Model(inputs, x)
        return model

    def define_word_count_model(self) -> Model:

        # Number of features
        print(self.word_count_train_features.shape)
        input_dim = self.word_count_train_features.shape[1]
        inputs = Input(shape=(input_dim,), name='word_count_input')
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.2, input_shape=(60,))(x)
        x = Dense(128, activation='relu')(x)
        model = Model(inputs, x)
        return model

    def define_pos_count_model(self) -> Model:

        # Number of features
        print(self.pos_count_train_features.shape)
        input_dim = self.pos_count_train_features.shape[1]
        inputs = Input(shape=(input_dim,), name='pos_input')
        x = Dense(128, activation='relu')(inputs)
        model = Model(inputs, x)
        return model

    def define_word_embedd_model(self) -> Model:

        input_dim = self.word_embedd_train_features.shape[1]
        inputs = Input(shape=(input_dim,), name='word_embedd_input')

        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)

        model = Model(inputs, x)
        return model

    def configure_model(self):
        """ Configures the learning process of the model.
        Here we have to choose a suitable optimizer.
        """

        optimizer = Adadelta()  # (lr=self.learning_rate,
        # rho=self.rho, epsilon=None, decay=0.0)

        self.model.compile(loss=self.loss_function,
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def train_and_plot_dl_model(self):
        """ Trains and plots the results of the current model.
        """
        train_data = []
        eval_data = []
        if self.use_stylometric:
            train_data.append(self.stylometric_train_features)
            eval_data.append(self.stylometric_eval_features)
        if self.use_rhyme:
            train_data.append(self.rhyme_train_features)
            eval_data.append(self.rhyme_eval_features)
        if self.use_word_count:
            train_data.append(self.word_count_train_features)
            eval_data.append(self.word_count_eval_features)
        if self.use_pos_count:
            train_data.append(self.pos_count_train_features)
            eval_data.append(self.pos_count_eval_features)
        if self.use_word_embedd:
            train_data.append(self.word_embedd_train_features)
            eval_data.append(self.word_embedd_eval_features)

        checkpoint_callback = ModelCheckpoint(
            f'{self.model_name}_checkpoint.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        metrics_test_handler = Metrics()
        metrics_test_handler.initialise_epoch_writer(f'{self.model_name}')

        metrics_eval_handler = Metrics()
        metrics_eval_handler.initialise_epoch_writer(f'{self.model_name}_eval')

        for e in range(self.epochs):
            print(f'Epoch {e} of {self.epochs}')
            history = self.model.fit(train_data, self.train_labels,
                                     batch_size=self.batch_size,
                                     epochs=1,
                                     validation_data=(
                                         eval_data, self.eval_labels),
                                     verbose=1,
                                     callbacks=[checkpoint_callback])

            predictions = self.predict_test_results()
            metrics_test_handler.calculate_metrics(
                predictions, self.test_labels)
            metrics_test_handler.write_epoch_metrics()

            predictions = self.predict_evaluation_results()
            metrics_eval_handler.calculate_metrics(
                predictions, self.encoder.inverse_transform(self.eval_labels))
            metrics_eval_handler.write_epoch_metrics()

    def plot_model_to_file(self, name: str, history):
        """Plots and saves the model to file.

        Arguments:
            name {[str]} -- used to create the file name
            history {[obj]} -- history object after training
        """
        plot_model(self.model, to_file=f'{name}_model.png', show_shapes=False,
                   show_layer_names=True, rankdir='TB')

    def plot_history(self, history, model_name):
        """Plots the achieved accuracy of the train and valid set

        Arguments:
            history {[type]} -- history object after training
            model_name {[type]} -- used to create the file name
        """

        print("plot_show_history >")
        # summarize history for accuracy

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(f'Accuracy of model; {model_name}')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        # plt.show()

        plt.savefig(f'{model_name}.png')
        plt.close()

        print("plot_show_history <")

    def load_model_checkPoint(self, model_name: str):
        self.model.load_weights(f'{model_name}_checkpoint.hdf5')

    def predict_evaluation_results(self):
        eval_data = []
        if self.use_stylometric:
            eval_data.append(self.stylometric_eval_features)
        if self.use_rhyme:
            eval_data.append(self.rhyme_eval_features)
        if self.use_word_count:
            eval_data.append(self.word_count_eval_features)
        if self.use_pos_count:
            eval_data.append(self.pos_count_eval_features)
        if self.use_word_embedd:
            eval_data.append(self.word_embedd_eval_features)

        predi = self.model.predict(eval_data)
        eval_results = np.argmax(predi, axis=1)
        results = np_utils.to_categorical(
            eval_results, num_classes=self.eval_labels.shape[0], dtype='int')
        return self.encoder.inverse_transform(results)

    def predict_test_results(self):
        test_data = []
        if self.use_stylometric:
            test_data.append(self.stylometric_test_features)
        if self.use_rhyme:
            test_data.append(self.rhyme_test_features)
        if self.use_word_count:
            test_data.append(self.word_count_test_features)
        if self.use_pos_count:
            test_data.append(self.pos_count_test_features)
        if self.use_word_embedd:
            test_data.append(self.word_embedd_test_features)

        predi = self.model.predict(test_data)
        test_results = np.argmax(predi, axis=1)
        results = np_utils.to_categorical(
            test_results, num_classes=self.test_labels.shape[0], dtype='int')
        return self.encoder.inverse_transform(results)


def write_results_to_file(predictions, gold_labels):
    with open(f'results.txt', 'w+') as the_file:
        output = '\n'.join(predictions)
        the_file.writelines(output)

    with open(f'gold.txt', 'w+') as the_file:
        output = '\n'.join(gold_labels)
        the_file.writelines(output)


def print_help():
    """ Prints the help message. """
    print('\nUsage:\n')
    print('-run <feature-directory-path> <output-directory> <model-name>- trains the model with the features of the given feature directory and writes its results to the output directory')
    print('-h or -help or --help    - print this message')


if __name__ == "__main__":

    print(len(sys.argv))
    if (len(sys.argv) != 4) or (sys.argv[1] == "-h" or sys.argv[1] == "-help" or sys.argv[1] == "--help"):
        print_help()
        sys.exit(1)

    print(sys.argv)
    feature_dir = sys.argv[1] + "/"
    output_dir = sys.argv[2] + "/"
    model_name = sys.argv[3]

    print(device_lib.list_local_devices())

    net = Net()
    dir = "stylometric/"
    net.stylometric_eval_features_path = feature_dir + dir + 'eval_features.npz'
    net.stylometric_train_features_path = feature_dir + dir + 'train_features.npz'
    net.stylometric_test_features_path = feature_dir + dir + 'test_features.npz'
    dir = "rhyme/"
    net.rhyme_eval_features_path = feature_dir + dir + 'eval_features.npz'
    net.rhyme_train_features_path = feature_dir + dir + 'train_features.npz'
    net.rhyme_test_features_path = feature_dir + dir + 'test_features.npz'
    dir = "word_count_vectors/"
    net.word_count_eval_features_path = feature_dir + dir + 'eval_features.npz'
    net.word_count_train_features_path = feature_dir + dir + 'train_features.npz'
    net.word_count_test_features_path = feature_dir + dir + 'test_features.npz'
    dir = "pos_count_vectors/"
    net.pos_count_eval_features_path = feature_dir + dir + 'eval_features.npz'
    net.pos_count_train_features_path = feature_dir + dir + 'train_features.npz'
    net.pos_count_test_features_path = feature_dir + dir + 'test_features.npz'
    dir = "word_embeddings/"
    net.word_embedd_eval_features_path = feature_dir + dir + 'eval_features.npz'
    net.word_embedd_train_features_path = feature_dir + dir + 'train_features.npz'
    net.word_embedd_test_features_path = feature_dir + dir + 'test_features.npz'

    net.train_labels_path = feature_dir + dir + 'train_labels.pkl'
    net.eval_labels_path = feature_dir + dir + 'eval_labels.pkl'
    net.test_labels_path = feature_dir + dir + 'test_labels.pkl'

    net.load_data()
    net.load_test_data()
    net.encode_labels()

    # define model hyper parameters
    net.batch_size = 64
    #net.learning_rate = 1.0
    net.epochs = 250
    net.loss_function = 'categorical_crossentropy'

    net.use_stylometric = True
    net.use_rhyme = True
    net.use_word_count = True
    net.use_pos_count = True
    net.use_word_embedd = True

    net.model_name = model_name
    net.build_concatenated_model()
    net.configure_model()

    print(net.model.summary())
    # plot graph
    plot_model(net.model, to_file='multilayer_perceptron_graph.png')
    net.train_and_plot_dl_model()

    net.load_model_checkPoint(net.model_name)

    metrics_handler = Metrics()
    metrics_handler.initialise_best_result_writer(net.model_name)

    # # write best result to file
    print("Best evaluation results:")
    predictions = net.predict_evaluation_results()
    metrics_handler.calculate_metrics(
        predictions, net.encoder.inverse_transform(net.eval_labels))
    metrics_handler.write_best_result(
        net.model_name, "checkpoint_on_eval_data")
    metrics_handler.print_result_scores()

    print("Model on test:")
    predictions = net.predict_test_results()
    metrics_handler.calculate_metrics(predictions, net.test_labels)
    metrics_handler.write_best_result(
        net.model_name, "checkpoint_on_test_data")
    metrics_handler.print_result_scores()
    write_results_to_file(predictions, net.test_labels)
