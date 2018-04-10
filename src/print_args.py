import sys
import argparse
import torch

from model_utils import print_args

data = torch.load(sys.argv[1])
print('Previous training info:')
print('Epoch: %d (start from 0)  Current iter: %d  Total iter: %d' % (data['aux']['epoch'], data['aux']['cur_iter'], data['aux']['total_iter']))
print(print_args(data['args']))
