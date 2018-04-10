import sys
import os
import argparse
import pickle

from holstep_parser import graph_from_hol_stmt

parser = argparse.ArgumentParser(description='Test utility for parser')
parser.add_argument('path', type=str, help='Path to dataset in graph format')
args = parser.parse_args()


class StatRecorder:
    def __init__(self):
        self.total_edge = 0
        self.longest = 0
        self.shortest = float('inf')
        self.item_count = 0
        self.total_node = 0
        self.max_node = 0
        self.min_node = float('inf')

    def count_edges(self, graph):
        total = 0
        for node in graph:
            total += len(node.outgoing)
        self.item_count += 1
        self.longest = max(self.longest, total)
        self.shortest = min(self.shortest, total)
        self.total_edge += total

    def count_node(self, graph):
        self.total_node += len(graph)
        self.max_node = max(self.max_node, len(graph))
        self.min_node = min(self.min_node, len(graph))


recorder = StatRecorder()

files = os.listdir(args.path)
for i, fname in enumerate(files):
    fpath = os.path.join(args.path, fname)
    print('Processing file {}/{} at {}.'.format(i + 1, len(files), fpath))
    with open(fpath, 'rb') as f:
        content = pickle.load(f)
        for pairs in content:
            recorder.count_edges(pairs[1])
            recorder.count_edges(pairs[2])
            recorder.count_node(pairs[1])
            recorder.count_node(pairs[2])

print('Statement stat: Max edges: {}\nMin edges: {}\nAvg edges: {}'.format(
    recorder.longest, recorder.shortest, recorder.total_edge / recorder.item_count))

print('Statement stat: Max nodes: {}\nMin nodes: {}\nAvg nodes: {}'.format(
    recorder.max_node, recorder.min_node, recorder.total_node /
    recorder.item_count))
