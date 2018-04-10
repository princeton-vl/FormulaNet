import sys
import os
import argparse

from holstep_parser import FormulaToGraph
from holstep_parser import parse_formula
from formula import NodeType


def check_incoming_outgoing(graph):
    '''This function checks if incoming is consistent with outgoing'''
    incoming_dict = {}
    outgoing_dict = {}
    for node in graph:
        if node.id in incoming_dict:
            assert False, 'Replicated id!'
        incoming_dict[node.id] = {x.id for x in node.incoming}
        outgoing_dict[node.id] = {x.id for x in node.outgoing}

    for node in graph:
        for x in node.incoming:
            assert node.id in outgoing_dict[x.id]
        for x in node.outgoing:
            assert node.id in incoming_dict[x.id]


def check_special_field(graph):
    '''Check special fields'''
    for node in graph:
        if node.type not in (NodeType.VAR, NodeType.VARFUNC):
            assert node.quant == None
        if node.type != NodeType.QUANT:
            assert len(node.vfunc) == 0 and node.vvalue is None


parser = argparse.ArgumentParser(description='Test utility for parser')
parser.add_argument('paths', type=str, help='Path for test files', nargs='+')
args = parser.parse_args()

converter = FormulaToGraph()

for path in args.paths:
    if os.path.isdir(path):
        files = os.listdir(path)
        for i, fname in enumerate(files):
            fpath = os.path.join(path, fname)
            print('Processing file {}/{} at {}.'.format(i + 1, len(files), fpath))
            with open(fpath, 'r') as f:
                for line in f:
                    if line and line[0] in '+-CA':
                        try:
                            token = next(f)
                            parsed_f = parse_formula(line[2:], token[2:])
                            graph = converter.convert(parsed_f)
                            check_incoming_outgoing(graph)
                            check_special_field(graph)
                        except (AssertionError, ValueError) as e:
                            print('----------------------')
                            print(line)
                            print(token)
                            raise e
