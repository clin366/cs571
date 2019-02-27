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

# from src.util import tsv_reader
from util import tsv_reader

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, concatenate, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,model_from_json
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.regularizers import l2


class SentimentAnalyzer(Component):
    def __init__(self, resource_dir: str, embedding_file='fasttext-50-180614.bin'):
        """
        Initializes all resources and the model.
        :param resource_dir: a path to the directory where resource files are located.
        """
        self.vsm = FastText(os.path.join(resource_dir, embedding_file))
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
        # serialize model to JSON
        # make sure directory exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_json = self.model.to_json()
        with open(model_path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_path + "/model.h5")
        print("Saved model to disk")
        # pass

    def padding_training(self, trn_xs, max_sentence_length = 80):

        blank_embedding = self.vsm.emb_list(' ')[0]
        train_xs = []
        for line in trn_xs:
            padding = max_sentence_length - len(line)
            for i in range(0, padding):
                line.append(blank_embedding)
            train_xs.append(line)
            
        train_xs = np.array(train_xs)
        train_xs = train_xs.reshape(train_xs.shape[0], train_xs.shape[1], train_xs.shape[2], 1)
        return train_xs

    def train(self, trn_data: List[Tuple[int, List[str]]], dev_data: List[Tuple[int, List[str]]], *args, **kwargs):
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

        # generate label vector
        number_of_classes = 5

        Y_train = np_utils.to_categorical(trn_ys, number_of_classes)
        Y_dev = np_utils.to_categorical(dev_ys, number_of_classes)

        # padding the sentence and generate training/developing dataset
        train_xs = self.padding_training(trn_xs)
        devlop_xs = self.padding_training(dev_xs)

        #Define the model
        first_ksize = 3
        second_ksize = 4
        third_ksize = 5
        max_sentence_length = 80
        embedding_dim = train_xs.shape[2]
        # instantiate regularizer
        reg = l2(0.15)
        image_input = Input(shape=(max_sentence_length,embedding_dim, 1))

        first_kernel = Conv2D(64, (first_ksize, embedding_dim),strides=(1, 1),padding='valid', activation = 'relu')(image_input)
        first_kernel = MaxPooling2D(pool_size=(max_sentence_length-first_ksize+1, 1), strides=(1,1), padding='valid')(first_kernel)
        first_kernel = Flatten()(first_kernel)
        # first_kernel 
        second_kernel = Conv2D(64, (second_ksize, embedding_dim),strides=(1, 1),padding='valid', activation = 'relu')(image_input)
        second_kernel = MaxPooling2D(pool_size=(max_sentence_length-second_ksize+1, 1), strides=(1,1), padding='valid')(second_kernel)
        second_kernel = Flatten()(second_kernel)
        # second_kernel
        third_kernel = Conv2D(64, (third_ksize, embedding_dim),strides=(1, 1),padding='valid', activation = 'relu')(image_input)
        third_kernel = MaxPooling2D(pool_size=(max_sentence_length-third_ksize+1, 1), strides=(1,1), padding='valid')(third_kernel)
        third_kernel = Flatten()(third_kernel)
        # third_kernel 
        merged = concatenate([first_kernel, second_kernel, third_kernel])
        merged = Dropout(0.5)(merged)
        output = Dense(5, activation='softmax', activity_regularizer=reg)(merged)

        self.model = Model(inputs=[image_input], outputs=output)
        # compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        # batch input
        gen = ImageDataGenerator()
        test_gen = ImageDataGenerator()
        train_generator = gen.flow(train_xs, Y_train, batch_size=50)
        test_generator = test_gen.flow(devlop_xs,Y_dev, batch_size = 50)
        # fit the model
        self.model.fit_generator(train_generator, steps_per_epoch=train_xs.shape[0]//50, epochs=15, 
                            validation_data=test_generator, validation_steps=devlop_xs.shape[0]//50)

        # pass

    def decode(self, data: List[Tuple[int, List[str]]], **kwargs) -> List[int]:
        """
        :param data:
        :param kwargs:
        :return: the list of predicted labels.
        """
        xs = [self.vsm.emb_list(x) for _, x in data]
        # TODO: to be filled
        padding_xs = self.padding_training(xs)
        pred = self.model.predict(padding_xs)
        y_classes = pred.argmax(axis=-1)
        return y_classes

    def evaluate(self, data: List[Tuple[int, List[str]]], **kwargs) -> float:
        """
        :param data:
        :param kwargs:
        :return: the accuracy of this model.
        """
        gold_labels = [y for y, _ in data]
        auto_labels = self.decode(data)
        total = correct = 0
        for gold, auto in zip(gold_labels, auto_labels):
            if gold == auto:
                correct += 1
            total += 1
        return 100.0 * correct / total


if __name__ == '__main__':
    resource_dir = os.environ.get('RESOURCE')
    sentiment_analyzer = SentimentAnalyzer(resource_dir)
    trn_data = tsv_reader(resource_dir, 'sst.trn.tsv')
    dev_data = tsv_reader(resource_dir, 'sst.dev.tsv')
    tst_data = tsv_reader(resource_dir, 'sst.tst.tsv')
    sentiment_analyzer.train(trn_data, dev_data)
    sentiment_analyzer.evaluate(tst_data)
    sentiment_analyzer.save(os.path.join(resource_dir, 'hw2-model'))
