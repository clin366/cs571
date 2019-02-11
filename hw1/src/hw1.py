# ========================================================================
# Copyright 2019 ELIT
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
import csv
import os
import glob
from typing import List, Any
import math

from elit.component import Component

__author__ = "Chen Lin, Gary Lai, Jinho D. Choi"


class HashtagSegmenter(Component):

    def __init__(self, resource_dir: str):
        """
        :param resource_dir: a path to the directory where resource files are located.
        """
        # initialize the n-grams
        # ngram_filenames = glob.glob(os.path.join(resource_dir, '[1-6]gram.txt'))
        one_gram_file = glob.glob(os.path.join(resource_dir, '1gram.txt'))[0]
        two_gram_file = glob.glob(os.path.join(resource_dir, '2gram.txt'))[0]
        # print(one_gram_file)
        # print(two_gram_file)
        # TODO: initialize resources
        self.gram1_dict = {}
        self.gram2_dict = {}
        
        # set the maximum word length to 18
        self.max_word_length = 18
        # record total word number
        self.total_words = 0
        
        self.read_dict(one_gram_file, two_gram_file)
#         pass
    
    def read_dict(self, one_gram, two_gram):
        """
        :param one_gram: file path for one_gram (e.g. 1gram.txt).
        :param two_gram: file path for two_gram (e.g. 2gram.txt).
        """
        gram1_dict_count = {}
        with open(one_gram, 'r') as fi:
            line = fi.readline().strip()
            while line:
                value, key = line.split('\t')
                gram1_dict_count[key] = float(value)
                line = fi.readline().strip()
                
        self.total_words = sum(gram1_dict_count.values()) 
        
        for key in gram1_dict_count.keys():
            self.gram1_dict[key] = math.log(gram1_dict_count[key]/self.total_words)
        
        with open(two_gram, 'r') as fi:
            line = fi.readline().strip()
            while line:
                value, key = line.split('\t')
                first_word, second_word = key.split(' ')
                
                if first_word in gram1_dict_count:
                    self.gram2_dict[key] = math.log(float(value)/gram1_dict_count[first_word])
                elif second_word in self.gram1_dict:
                    self.gram2_dict[key] = self.gram1_dict[second_word]
                line = fi.readline().strip()

    def decode(self, hashtag: str, **kwargs) -> List[str]:
        """
        :param hashtag: the input hashtag starting with `#` (e.g., '#helloworld').
        :param kwargs:
        :return: the list of tokens segmented from the hashtag (e.g., ['hello', 'world']).
        """
        # TODO: update the following code.
        hashtag = hashtag[1:]
        sequence = hashtag.lower()

        # Do dynamic programming to find the maximum probability segmentation
        word_list = self.dynamic_get_best_nodes(sequence)
        
        # Restore the original upper/lower case
        index = 0
        real_seg = []
        for word in word_list:
            real_seg.append(hashtag[index:index+len(word)])
            index += len(word)
        return real_seg
    
    def dynamic_get_best_nodes(self, sequence):
        """
        :param sequence: the lowercase of input hashtag starting (e.g., 'helloworld').
        :return: the list of tokens segmented from the sequence (e.g., ['hello', 'world']).
        """
        nodes_list = [{'pre_node':-1, 'freq_sum':0}] 
        
        for node in range(1,len(sequence) + 1):
            (max_pre_node, max_freq_sum) = self.get_max_pre_node(sequence, node, nodes_list)
            nodes_list.append({'pre_node':max_pre_node, 'freq_sum':max_freq_sum})
            
        best_nodes_list = []
        node = len(sequence) 
        best_nodes_list.append(node)
        pre_node= nodes_list[node]["pre_node"]
        
        while (pre_node != -1):
            best_nodes_list.append(pre_node)
            pre_node = nodes_list[pre_node]["pre_node"]
        best_nodes_list.reverse()

        word_list = []
        for i in range(len(best_nodes_list)-1):
            word = sequence[best_nodes_list[i]:best_nodes_list[i + 1]]
            word_list.append(word)
        return word_list
    
    def get_word_freq(self, word):
        if word in self.gram1_dict: 
            prob = self.gram1_dict[word]
        else:
            prob = math.log(10./(self.total_words*(10**len(word))))
        return prob
    
    def get_word_trans_freq(self, first_word, second_word):
        trans_word =  first_word + " " + second_word
        if trans_word in self.gram2_dict:
            trans_prob = self.gram2_dict[trans_word]
        else:
            trans_prob = self.get_word_freq(second_word)
        return trans_prob
    
    def get_max_prob_node(self, node_list):
        max_prob = -999999.
        max_node = None
        for node in node_list:
            if (node[1] > max_prob):
                max_node = node
                max_prob = node[1]
        return max_node
    
    def get_max_pre_node(self, sequence, node, nodes_list):
        
        max_seg_length = min([node, self.max_word_length]) + 1
        pre_node_list = [] 
        
        for segment_length in range(1,max_seg_length):
            pre_node = node-segment_length
            segment = sequence[pre_node:node] 
            
            if pre_node == 0: 
                segment_prob = self.get_word_freq(segment)
            else: 
                pre_pre_word = sequence[nodes_list[pre_node]["pre_node"]:pre_node]
                segment_prob = self.get_word_trans_freq(pre_pre_word, segment)
            
            total_prob = nodes_list[pre_node]["freq_sum"] + segment_prob
            pre_node_list.append((pre_node, total_prob))
                                    
        return self.get_max_prob_node(pre_node_list)


    def evaluate(self, data: Any, **kwargs):
        pass  # NO NEED TO UPDATE

    def load(self, model_path: str, **kwargs):
        pass  # NO NEED TO UPDATE

    def save(self, model_path: str, **kwargs):
        pass  # NO NEED TO UPDATE

    def train(self, trn_data, dev_data, *args, **kwargs):
        pass  # NO NEED TO UPDATE


if __name__ == '__main__':
    resource_dir = os.environ.get('RESOURCE')
    segmenter = HashtagSegmenter(resource_dir)
    total = correct = 0

    with open(os.path.join(resource_dir, 'hashtags.csv')) as fin:
        reader = csv.reader(fin)
        for row in reader:
            hashtag = row[0]
            gold = row[1]
            auto = ' '.join(segmenter.decode(hashtag))
            print('%s -> %s | %s' % (hashtag, auto, gold))
            # print('auto:'),
            # print(auto)
            # print('gold:'),
            # print(gold)
            # print(auto == gold)
            if gold == auto: correct += 1
            total += 1

    print('%5.2f (%d/%d)' % (100.0*correct/total, correct, total))
