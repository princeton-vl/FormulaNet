import numpy as np
import time
import random
import torch.nn as nn
import torch
import math
import os
import sys
sys.path.insert(0, 'data_util')
import pickle
from torch.multiprocessing import Process, Queue
from torch.autograd import Variable
from formula import NodeType, Node
from utils import split_list

COMM_OP = {"/\\:c", "\\/:c", "+:c", "*:c", "==:c", "|-:c"}  # Special treatment for |-

# Input data format
# [onehot, index1, index2, mat]
# onehot: Tensor (num_nodes*input_dim) one hot from dict
# index1: Long Tensor (num_pairs) (source)
# index2: Long Tensor (num_pairs) (target)
# mat: Tensor for index add (num_nodes * num_pairs)


class DataLoader(object):
    def __init__(self,
                 formula_path,
                 dict_path,
                 separate_conj_stmt=False,
                 binary=False,
                 part_no=-1,
                 part_total=0,
                 file_list=None,
                 deepmath=False,
                 norename=False,
                 filter_abelian=False,
                 compatible=False):  # part_no, part_total: will not shuffle.
        self.formula_path = formula_path
        self.dict_path = dict_path
        self.maxsize = 500  # maxsize for async queue
        self.iter_ = 0  # epoch. Legacy reason for its name
        self.total_in_epoch = -1  # conj, stmt pairs supply in current epoch.
        self.total_iter = -1  # total iteration
        self.rename = not norename
        if not os.path.exists(dict_path):
            self.dict = self.build_dictionary()
        else:
            self.dict = torch.load(dict_path)
        self.queue = Queue(self.maxsize)
        self.reader = Process(target=self.read)
        self.dict_size = len(self.dict.keys())
        self.separate_conj_stmt = separate_conj_stmt
        self.binary = binary
        self.part_no = part_no
        self.part_total = part_total
        if file_list is None:
            file_list = os.listdir(self.formula_path)
            if part_total != 0:
                file_list.sort()
                file_list = split_list(file_list, part_total, part_no)
        else:
            if part_total != 0:
                file_list = split_list(file_list, part_total, part_no)
        self.file_list = file_list
        self.deepmath = deepmath
        self.filter_abelian = filter_abelian
        self.compatible = compatible

    def start_reader(self):
        self.reader.daemon = True
        self.reader.start()

    def next_batch(self):
        # [conjecture, statement, label, conj_binary, stmt_binary]
        data = self.queue.get()
        if data is None:
            self.iter_ += 1
            self.total_in_epoch = 0
        else:
            self.total_in_epoch += 1
            self.total_iter += 1
        return data

    def build_dictionary(self):
        def _deter_name(node):
            node_name = node.name
            if node.type == NodeType.VAR:
                node_name = 'VAR'
            elif node.type == NodeType.VARFUNC:
                node_name == 'VARFUNC'
            return node_name

        files = os.listdir(self.formula_path)
        tokens = set({})
        dicts = {}
        for i, a_file in enumerate(files):
            with open(os.path.join(self.formula_path, a_file), 'rb') as f:
                print('Loading file {}/{}'.format(i + 1, len(files)))
                dataset = pickle.load(f)
                for j, pair in enumerate(dataset):
                    print('Processing pair {}/{}'.format(j + 1, len(dataset)))
                    if self.rename:
                        tokens.update([_deter_name(x) for x in pair[1]])
                        tokens.update([_deter_name(x) for x in pair[2]])
                    else:
                        tokens.update([x.name for x in pair[1]])
                        tokens.update([x.name for x in pair[2]])

        for i, x in enumerate(tokens):
            dicts[x] = i
        dicts['UNKNOWN'] = len(dicts)
        if 'VAR' not in dicts:
            dicts['VAR'] = len(dicts)
        if 'VARFUNC' not in dicts:
            dicts['VARFUNC'] = len(dicts)
        torch.save(dicts, self.dict_path)
        return dicts
    
    def _decide_name(self, node):
        node_name = node.name
        if self.rename:
            if node.type == NodeType.VAR:
                node_name = 'VAR'
            elif node.type == NodeType.VARFUNC:
                node_name == 'VARFUNC'

        if node_name not in self.dict:
            node_name = 'UNKNOWN'
        return node_name

    def generate_one_sentence(self, sentence):
        # Undirected graph
        # index1 starts, index2 ends
        index1 = []
        index2 = []
        onehot_collect = []
        id2pos = {node.id: i for i, node in enumerate(sentence)}

        for i, node in enumerate(sentence):
            for x in node.incoming:
                index1.append(id2pos[x.id])
                index2.append(id2pos[node.id])
            for x in node.outgoing:
                index1.append(id2pos[x.id])
                index2.append(id2pos[node.id])

            node_name = self._decide_name(node)
            onehot_collect.append(self.dict[node_name])

        index1 = np.array(index1)
        index2 = np.array(index2)
        mat = np.zeros((len(sentence), len(index2)), dtype=np.float32)
        for x in sentence:
            mat[id2pos[x.id], index2 == id2pos[x.id]] = 1.0 / np.sum(
                index2 == id2pos[x.id])

        if self.compatible:
            onehot = np.zeros((len(sentence), self.dict_size), dtype=np.float32)
            onehot[range(len(sentence)), onehot_collect] = 1

        index1 = torch.from_numpy(index1)
        index2 = torch.from_numpy(index2)
        if self.compatible:
            onehot = torch.from_numpy(onehot)
        else:
            onehot = torch.LongTensor(onehot_collect)
        mat = torch.from_numpy(mat)

        return (onehot, index1, index2, mat)

    def directed_generate_one_sentence(self, sentence):
        # Distinguish in-edges and out-edges
        # index1 starts, index2 ends
        iindex1 = []
        iindex2 = []
        oindex1 = []
        oindex2 = []
        id2pos = {node.id: i for i, node in enumerate(sentence)}
        onehot_collect = []

        for node in sentence:
            for x in node.incoming:
                iindex1.append(id2pos[x.id])
                iindex2.append(id2pos[node.id])
            for x in node.outgoing:
                oindex1.append(id2pos[node.id])
                oindex2.append(id2pos[x.id])

            node_name = self._decide_name(node)
            onehot_collect.append(self.dict[node_name])

        # Incoming
        iindex1 = np.array(iindex1)
        iindex2 = np.array(iindex2)
        oindex1 = np.array(oindex1)
        oindex2 = np.array(oindex2)
        imat = np.zeros((len(sentence), len(iindex2)), dtype=np.float32)
        omat = np.zeros((len(sentence), len(oindex1)), dtype=np.float32)

        for x in sentence:
            imat[id2pos[x.id], iindex2 == id2pos[x.id]] = 1.0 / (
                np.sum(oindex1 == id2pos[x.id]) + np.sum(iindex2 == id2pos[x.id]))

        # Outgoing
        for x in sentence:
            omat[id2pos[x.id], oindex1 == id2pos[x.id]] = 1.0 / (
                np.sum(oindex1 == id2pos[x.id]) + np.sum(iindex2 == id2pos[x.id]))

        if self.compatible:
            onehot = np.zeros((len(sentence), self.dict_size), dtype=np.float32)
            onehot[range(len(sentence)), onehot_collect] = 1

        iindex1 = torch.from_numpy(iindex1)
        iindex2 = torch.from_numpy(iindex2)
        oindex1 = torch.from_numpy(oindex1)
        oindex2 = torch.from_numpy(oindex2)
        if self.compatible:
            onehot = torch.from_numpy(onehot)
        else:
            onehot = torch.LongTensor(onehot_collect)
        imat = torch.from_numpy(imat)
        omat = torch.from_numpy(omat)

        return (onehot, iindex1, iindex2, imat, oindex1, oindex2, omat)

    def generate_one_sentence_binary(self, sentence):
        # directed graph
        index = []
        id2pos = {node.id: i for i, node in enumerate(sentence)}
        for node in sentence:
            if len(node.outgoing) > 1 and not (self.filter_abelian and node.name in COMM_OP):
                for i, n1 in enumerate(node.outgoing):
                    for n2 in node.outgoing[i + 1:]:
                        index.append(id2pos[node.id])
                        index.append(id2pos[n1.id])
                        index.append(id2pos[n2.id])
            if len(node.outgoing) > 1 and (self.filter_abelian and node.name == '|-:c'):
                for n1 in node.outgoing[1:]:
                    index.append(id2pos[node.id])
                    index.append(id2pos[node.outgoing[0].id])
                    index.append(id2pos[n1.id])
        index = np.array(index)
        mat = np.zeros((len(sentence), len(index)), dtype=np.float32)
        for x in sentence:
            f = index == id2pos[x.id]
            if np.sum(f) > 0:
                mat[id2pos[x.id], f] = 1.0 / np.sum(f)
        #print (index.shape, mat.shape)
        if index.shape[0] > 0:
            return (torch.from_numpy(index.reshape(-1, 3).T), torch.from_numpy(mat))
        else:
            #print (index.shape, mat.shape)
            return (torch.Tensor(1), torch.Tensor(1))

    def read(self):
        files = self.file_list
        while True:
            random.shuffle(files)
            for a_file in files:
                with open(os.path.join(self.formula_path, a_file), 'rb') as f:
                    content = pickle.load(f)
                    random.shuffle(content)
                    for x in content:
                        flag, conj, stmt = x
                        if self.separate_conj_stmt:
                            self.queue.put(
                                (self.directed_generate_one_sentence(conj),
                                 self.directed_generate_one_sentence(stmt), flag))
                        elif self.binary:
                            self.queue.put(
                                (self.directed_generate_one_sentence(conj),
                                 self.directed_generate_one_sentence(stmt), flag,
                                 self.generate_one_sentence_binary(conj),
                                 self.generate_one_sentence_binary(stmt)))
                        else:
                            self.queue.put((self.generate_one_sentence(conj),
                                            self.generate_one_sentence(stmt), flag))
            self.queue.put(None)

    def destruct(self):
        self.reader.terminate()
