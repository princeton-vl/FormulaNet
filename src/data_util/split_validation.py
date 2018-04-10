import os
import shutil
import random
import argparse


parser = argparse.ArgumentParser(
    description='Generate graph repr dataset from HolStep')
parser.add_argument('--data', type=str, required=True, help='Path to unsplitted training set')
parser.add_argument('--train', type=str, required=True, help='Path to splitted training')
parser.add_argument('--valid', type=str, required=True, help='Path to splitted validation')
parser.add_argument('--vlist', type=str, help='List of the files for validation set.')
parser.add_argument('--partition', '-p', type=float, help='Percentage of validation. Only work when vlist is not specified', default=0.07)

args = parser.parse_args()


files = os.listdir(args.data)
if args.vlist is None:
    valid_files = random.sample(files, int(len(files)*args.partition+0.5))
else:
    with open(args.vlist, 'r') as f:
        valid_files = f.readlines()
        valid_files = [x.strip() for x in valid_files]
    print('\nUse provided list: {}'.format(args.vlist))

for i, fname in enumerate(files):
    print('Copying file {}/{}'.format(i, len(files)))
    shutil.copy(os.path.join(args.data, fname), args.train)

for j, fname in enumerate(valid_files):
    print('Moving file {}/{}'.format(j, len(valid_files)))
    shutil.move(os.path.join(args.train, fname), args.valid)

train_files = os.listdir(args.train)
valid_files = os.listdir(args.valid)
print('\nTraining set count: {}\nValidation set count: {}'.format(len(train_files), len(valid_files)))
