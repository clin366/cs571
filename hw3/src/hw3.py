# ========================================================================
# Copyright 2019 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import os
from typing import List, Tuple

from elit.component import Component
from elit.embedding import FastText
from elit.eval import ChunkF1

from src.util import tsv_reader

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, concatenate, Dropout, Activation, Flatten
from keras.layers import TimeDistributed,LSTM,Bidirectional
from keras.models import Model,model_from_json
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.regularizers import l2
import math
import json


class NamedEntityRecognizer(Component):
    def __init__(self, resource_dir: str, embedding_file='fasttext-50-180614.bin'):
        """
        Initializes all resources and the model.
        :param resource_dir: a path to the directory where resource files are located.
        """
        self.vsm = FastText(os.path.join(resource_dir, embedding_file))

        if os.path.exists(resource_dir + '/label2Idx.json'):
            with open(resource_dir + '/label2Idx.json') as fi:
                self.label2Idx = json.load(fi)

            self.idx2Label = {v: k for k, v in self.label2Idx.items()}


        # TODO: to be filled.

    def load(self, model_path: str, **kwargs):
        """
        Load the pre-trained model.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        json_file = open(model_path + '/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(model_path + "/model.h5")
        print("Loaded model from disk")
        # pass

    def save(self, model_path: str, **kwargs):
        """
        Saves the current model to the path.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_json = self.model.to_json()
        with open(model_path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_path + "/model.h5")
        print("Saved model to disk")
        pass

    def train(self, trn_data: List[Tuple[List[str], List[str]]], dev_data: List[Tuple[List[str], List[str]]], *args, **kwargs):
        """
        Trains the model.
        :param trn_data: the training data.
        :param dev_data: the development data.
        :param args:
        :param kwargs:
        :return:
        """
        trn_ys, trn_xs = zip(*[(y, self.vsm.emb_list(x)) for y, x in trn_data])
        dev_ys, dev_xs = zip(*[(y, self.vsm.emb_list(x)) for y, x in dev_data])
        # TODO: to be filled

        labelSet = set()

        for dataset in [trn_ys, dev_ys]:
            for label_seq in dataset:
                for label in label_seq:
                    labelSet.add(label)

        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)
        self.label2Idx['PAD'] = len(self.label2Idx)

        with open(resource_dir + '/label2Idx.json', 'w') as fo:
            json.dump(self.label2Idx, fo)

        self.idx2Label = {v: k for k, v in self.label2Idx.items()}

        Y_train = self.createMatrices(trn_ys, self.label2Idx)
        Y_dev = self.createMatrices(dev_ys, self.label2Idx)

        train_xs = self.padding_training(trn_xs)
        devlop_xs = self.padding_training(dev_xs)

        Y_train_padding = self.padding_training_Y(Y_train)
        Y_dev_padding = self.padding_training_Y(Y_dev)

        train_ys = [np_utils.to_categorical(i, num_classes=len(self.label2Idx)) for i in Y_train_padding]
        train_ys = np.asarray(train_ys)

        devlop_ys = [np_utils.to_categorical(i, num_classes=len(self.label2Idx)) for i in Y_dev_padding]
        devlop_ys = np.asarray(devlop_ys)

        max_sentence_length = 113
        embedding_dim = train_xs.shape[2]

        image_input = Input(shape=(max_sentence_length,embedding_dim))

        output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(image_input)
        output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)

        self.model = Model(inputs=[image_input], outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='nadam')

        self.model.fit(train_xs, train_ys, batch_size=50, epochs = 15, validation_data=(devlop_xs, devlop_ys))

    def createMatrices(self, sentences, label2Idx):
        dataset = []
        for sentence in sentences:
            labelIndices = []
            for label in sentence:  
                labelIndices.append(self.label2Idx[label])
               
            dataset.append(labelIndices) 
        return dataset

    def padding_training(self, trn_xs, max_sentence_length = 113):

        blank_embedding = self.vsm.emb_list(' ')[0]
        train_xs = []
        for line in trn_xs:
            padding = max_sentence_length - len(line)
            for i in range(0, padding):
                line.append(blank_embedding)
            train_xs.append(line)

        train_xs = np.array(train_xs)
        train_xs = train_xs.reshape(train_xs.shape[0], train_xs.shape[1], train_xs.shape[2])
        return train_xs

    def padding_training_Y(self, Y_train, max_sentence_length = 113):

        
        blank_embedding = len(self.label2Idx) - 1 
        Y_train_padding = []
        for line in Y_train:
            padding = max_sentence_length - len(line)
            for i in range(0, padding):
                line.append(blank_embedding)
            Y_train_padding.append(line)
        
        Y_train_padding = np.array(Y_train_padding)
        return Y_train_padding

        # pass

    def decode(self, data: List[Tuple[List[str], List[str]]], **kwargs) -> List[str]:
        """
        :param data:
        :param kwargs:
        :return: the list of predicted labels.
        """
        xs = [self.vsm.emb_list(x) for _, x in data]
        # TODO: to be filled

        padding_xs = self.padding_training(xs)
        padding_pred = self.model.predict(padding_xs)

        padding_idx = np.argmax(padding_pred, axis=2) 
        padding_labels = []
        for sentences in padding_idx:
            token_label = []
            for l in sentences:
                if l != len(self.label2Idx) - 1:
                    token_label.append(self.idx2Label[l])
            padding_labels.append(token_label)

        # y_classes = pred.argmax(axis=-1)
        return padding_labels

    def evaluate(self, data: List[Tuple[List[str], List[str]]], **kwargs) -> float:
        """
        :param data:
        :param kwargs:
        :return: the accuracy of this model.
        """
        preds = self.decode(data)
        labels = [y for y, _ in data]
        acc = ChunkF1()
        for pred, label in zip(preds, labels):
            acc.update(pred, label)
        return float(acc.get()[1])


if __name__ == '__main__':
    resource_dir = os.environ.get('RESOURCE')
    sentiment_analyzer = NamedEntityRecognizer(resource_dir)
    trn_data = tsv_reader(resource_dir, 'conll03.eng.trn.tsv')
    dev_data = tsv_reader(resource_dir, 'conll03.eng.dev.tsv')
    tst_data = tsv_reader(resource_dir, 'conll03.eng.tst.tsv')
    sentiment_analyzer.train(trn_data, dev_data)
    sentiment_analyzer.evaluate(tst_data)
    sentiment_analyzer.save(os.path.join(resource_dir, 'hw3-model'))
