"Draw Larry's favorite graph!"
from graphviz import Digraph


import numpy as np
import time
import random
import torch.nn as nn
import torch
import math
import os
import sys
import pickle
sys.path.insert(0, 'data_util')
from torch.multiprocessing import Process, Queue
from torch.autograd import Variable
from formula import NodeType, Node
from utils import split_list

def draw(node_list, graph_name):
    dot = Digraph(comment=graph_name)
    for node in node_list:
        dot.node(str(node.id), node.name)
    for node in node_list:
        for out in node.outgoing:
            dot.edge(str(node.id),str(out.id))

    dot.render(graph_name, view=True)
