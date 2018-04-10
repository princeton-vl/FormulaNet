'''
The commandline settings can be displayed by passing a --help flag
To run any model, --data_path and --dict_file need to be set.
--dict_file can be automatically generated if no file is detected on
the given path (you just need to give a path for saving the dict)

if --record <file_path> is provided, a history torch.save object will be saved.

Implicit assumption: in --data_path there are two folders, train and valid.

[WARNING]
If you use --resume, it is your responsibility to specify a compatible
set of arguments
'''
import os
import sys
import numpy as np
import math
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe
from torch.autograd import Variable

import data_loader
import model
import loss
from model_utils import save_model
from model_utils import load_model
from model_utils import get_opt
from model_utils import print_args
from model_utils import HistoryRecorder
from model_utils import named_buffers
from model_utils import FakeModule

import log

def parse_args():
    # Implicit assumption:
    # Training data in train, validation data in valid
    parser = argparse.ArgumentParser(description='The ultimate optimizer')
    parser.add_argument('--nFeats', type=int, help='Number of features', default=256)
    parser.add_argument(
        '--nSteps', type=int, help='Number of step graphnet updates', default=3)
    parser.add_argument(
        '--learning_rate', '-l', type=float, help='Learning rate', default=0.001)
    parser.add_argument(
        '--min_lr', type=float, help='minimum learning rate', default=0.00002)
    parser.add_argument('--log', type=str, help='Path to log file', required=True)
    parser.add_argument(
        '--output', type=str, help='Path to model output', required=True)
    parser.add_argument(
        '--data_path', type=str, help='Path to data set', default='../data/hol_data/')
    parser.add_argument(
        '--dict_file',
        type=str,
        help='Path to dict cache',
        default='../data/dicts/new_hol_train_dict')
    parser.add_argument(
        '--observe', type=int, help='Output info of training status', default=20000)
    parser.add_argument(
        '--max_pair',
        type=int,
        help='Max pair cut-off line for batching',
        default=200000)
    parser.add_argument('--epoch', '-e', type=int, help='Number of epoch', default=5)
    parser.add_argument(
        '--optim',
        type=str,
        help='Choose optimizer from sgd, rmsprop and adam',
        default='rmsprop')
    parser.add_argument(
        '--voting', action='store_true', help='Voting or use the last step to predict')
    parser.add_argument(
        '--check_num',
        type=int,
        help='Number of samples for validation',
        default=950000)
    parser.add_argument(
        '--module_depth', type=int, help='number of layers in a block', default=2)
    parser.add_argument(
        '--bias', type=bool, help='bias in linear layers', default=False)
    parser.add_argument(
        '--uncondition',
        action='store_true',
        help='Flag to enable uncondition learning')
    parser.add_argument(
        '--block', type=str, help='Block used in graph net', default='normal')
    parser.add_argument(
        '--add_conj', action='store_true', help='Add conj in stmt graph')
    parser.add_argument(
        '--short_cut', action='store_false', help='Disable short Cut between steps')
    parser.add_argument(
        '--direction',
        action='store_false',
        help='Disable using different weights for in-edge and out-edge')
    parser.add_argument(
        '--record',
        type=str,
        help=
        'Save training/validation history by torch.save. Save only when this option is provided'
    )
    parser.add_argument(
        '--loss',
        type=str,
        help='Choose the type of the loss used in the network',
        default='concat')
    parser.add_argument('--worker', type=int, help='Number of workers', default=5)
    parser.add_argument(
        '--binary',
        action='store_false',
        help='Disable updating with each pair of outgoing edges with common sourse node.')
    parser.add_argument(
        '--epoch_len', type=int, help='Specify 1 epoch = ? iteration.', default=1900000)
    parser.add_argument(
        '--norename', action='store_true', help="Not rename for variables")
    parser.add_argument(
        '--fabelian', action='store_true', help='Filter commutative ops in binary mode')
    parser.add_argument(
        '--loss_layers', type=int, nargs='+', help='Decide layers in the loss')
    parser.add_argument(
        '--dropout', type=float, help='Dropout layer in the loss.', default=0)
    parser.add_argument(
        '--no_step_supervision', action='store_true', help='Do not use step supervision.')
    parser.add_argument(
        '--cond_short_cut', action='store_true', help='Add residual links in condloss')
    parser.add_argument(
        '--tied_weight', action='store_true', help='Use tied_weight for each step')
    parser.add_argument('--compatible', action='store_true', help='Use compatible mode to run.')
    parser.add_argument('--resume', type=str, help='resume training to given model')
    parser.add_argument('--resume_only_net', action='store_true', help='Resume only GraphNet')
    parser.add_argument('--fix_net', action='store_true', help='Fix GraphNet')
    parser.add_argument('--unfix_net_after', type=int, help='Unfix Graph after certain iteration')

    # options:
    # 1. concat: basic model. Concatenate maxpooled conj and maxpooled stmt
    # 2. concat_em_uc: Concatenate elem prod of maxpooled conj and maxpooled stmt and
    #       maxpooled stmt
    # 3. em: Elem prod of maxpooled conj and maxpooled stmt
    # 4. mixmax: ucloss + max of all pairs of dot products
    # 5. mixmean: ucloss + mean of all pairs of dot products
    # 6. condloss: Stronger condition
    # 7. mulloss: stronger similarity

    # Global argmuments
    args = parser.parse_args()

    setattr(args, 'train_file', os.path.join(args.data_path, 'train'))
    setattr(args, 'test_file', os.path.join(args.data_path, 'valid'))
    if args.no_step_supervision:
        loss_step = 1
    else:
        loss_step = args.nSteps + 1
    setattr(args, 'loss_step', loss_step)
    # unset directed if binary is set.
    if args.direction and args.binary:
        setattr(args, 'direction', False)
    return args


def main():
    args = parse_args()

    mp.set_start_method('spawn')  # Using spawn is decided.
    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))

    loaders = []
    file_list = os.listdir(args.train_file)
    random.shuffle(file_list)
    for i in range(args.worker):
        loader = data_loader.DataLoader(
            args.train_file,
            args.dict_file,
            separate_conj_stmt=args.direction,
            binary=args.binary,
            part_no=i,
            part_total=args.worker,
            file_list=file_list,
            norename=args.norename,
            filter_abelian=args.fabelian,
            compatible=args.compatible)
        loaders.append(loader)
        loader.start_reader()

    net, mid_net, loss_fn = create_models(args, loaders[0], allow_resume=True)
    # Use fake modules to replace the real ones
    net = FakeModule(net)
    if mid_net is not None:
        mid_net = FakeModule(mid_net)
    for i in range(len(loss_fn)):
        loss_fn[i] = FakeModule(loss_fn[i])

    opt = get_opt(net, mid_net, loss_fn, args)

    inqueues = []
    outqueues = []

    plist = []
    for i in range(args.worker):
        recv_p, send_p = Pipe(False)
        recv_p2, send_p2 = Pipe(False)
        inqueues.append(send_p)
        outqueues.append(recv_p2)
        plist.append(
            Process(target=worker, args=(recv_p, send_p2, loaders[i], args, i)))
        plist[-1].start()

    _logger.warning('Training begins')
    train(inqueues, outqueues, net, mid_net, loss_fn, opt, loaders, args, _logger)
    loader.destruct()
    for p in plist:
        p.terminate()
    for loader in loaders:
        loader.destruct()
    _logger.warning('Training ends')


def create_models(args, loader, allow_resume=False):
    net = model.GraphNet(loader.dict_size, args.nFeats, args.nSteps, args.block,
                         args.module_depth, args.bias, args.short_cut, args.direction,
                         args.loss, args.binary,
                         no_step_supervision=args.no_step_supervision,
                         tied_weight=args.tied_weight,
                         compatible=args.compatible).cuda()

    mid_net = None
    if args.loss in ('mixmax', 'mixmean'):
        mid_net = model.FullyConnectedNet(
            args.nFeats, args.nFeats // 2, bias=args.bias).cuda()

    loss_fn = []

    for i in range(args.loss_step):
        if args.loss == 'mulloss':
            loss_fn.append(loss.MultiplyLoss(args.nFeats,
                                             cond_short_cut=args.cond_short_cut).cuda())
        elif args.loss == 'condloss':
            loss_fn.append(loss.CondLoss(args.nFeats * 2, args.nFeats,
                                         layer_list=args.loss_layers,
                                         dropout=args.dropout,
                                         cond_short_cut=args.cond_short_cut).cuda())
        elif args.loss in ('concat', 'concat_em_uc'):
            if args.uncondition or args.add_conj:
                loss_fn.append(loss.ClassifyLoss(args.nFeats, args.nFeats // 2,
                                                 layer_list=args.loss_layers,
                                                 dropout=args.dropout,
                                                 bias=args.compatible).cuda())
            else:
                loss_fn.append(loss.ClassifyLoss(args.nFeats * 2, args.nFeats,
                                                 layer_list=args.loss_layers,
                                                 dropout=args.dropout,
                                                 bias=args.compatible).cuda())
        elif args.loss in ('mixmax', 'mixmean'):
            loss_fn.append(loss.UCSimLoss(args.nFeats, args.nFeats // 2,
                                          layer_list=args.loss_layers,
                                          dropout=args.dropout).cuda())
        elif args.loss == 'pair':
            loss_fn.append(loss.ClassifyLoss(args.nFeats // 2, args.nFeats // 4,
                                             layer_list=args.loss_layers,
                                             dropout=args.dropout).cuda())
        elif args.loss == 'em':
            loss_fn.append(loss.ClassifyLoss(args.nFeats, args.nFeats // 2,
                                             layer_list=args.loss_layers,
                                             dropout=args.dropout).cuda())
        else:
            assert False, 'Wrong --loss option!'

    if args.resume is not None and allow_resume:
        data = torch.load(args.resume)
        _logger = log.get_logger('Create Model', args)
        _logger.warning('Load Model!')
        _logger.info('Previous training info:')
        _logger.info('Epoch: %d  Current iter: %d  Total iter: %d', data['aux']['epoch'], data['aux']['cur_iter'], data['aux']['total_iter'])
        _logger.warning('Previous training args:')
        _logger.info(print_args(data['args']))
        net.load_state_dict(data['net']['state_dict'])
        if mid_net is not None:
            mid_net.load_state_dict(data['mid_net']['state_dict'])
        if not args.resume_only_net:
            for i in range(len(loss_fn)):
                loss_fn[i].load_state_dict(data['loss_fn'][i]['state_dict'])

    return net, mid_net, loss_fn


def add_conj(fact):
    l1 = fact[3].size()[0]
    l2 = fact[3].size()[1]
    index1_ = torch.cat((fact[2], torch.range(0, l1 - 1).long()), 0)
    index2_ = torch.cat((fact[1], torch.zeros(l1).long() + l1), 0)
    mat_ = torch.Tensor(l1, l1 + l2)
    a = 1 / (fact[3] + 1e-10) + 1
    mat_[:, l2:] = (1 / a.min(1)[0].resize_(l1)).diag()
    mat_[:, :l2] = 1 / a - 1e-10
    return [fact[0], index1_, index2_, mat_]


def embed_step(pair_total,
               net,
               mid_net,
               loss_fn,
               catfeat,
               label,
               loader_,
               args,
               volatile=False):
    data = loader_.next_batch()
    if data is None:
        epoch_end = True
        return pair_total, epoch_end
    if args.binary == False:
        conjecture, fact, y = data
    else:
        conjecture, fact, y, conj_bi, fact_bi = data
    x1 = None
    x2 = None
    if args.add_conj:
        fact = add_conj(fact)
    if args.direction:
        x1 = [
            Variable(fact[0].cuda(), volatile=volatile),
            fact[1].cuda(),
            fact[2].cuda(),
            Variable(fact[3].cuda(), volatile=volatile),
            fact[4].cuda(),
            fact[5].cuda(),
            Variable(fact[6].cuda(), volatile=volatile)
        ]  # yapf: disable
    elif args.binary:
        x1 = [
            Variable(fact[0].cuda(), volatile=volatile), fact[1].cuda(), fact[2].cuda(),
            Variable(fact[3].cuda(), volatile=volatile), fact[4].cuda(), fact[5].cuda(),
            Variable(fact[6].cuda(), volatile=volatile), fact_bi[0].cuda(), Variable(
                fact_bi[1].cuda(), volatile=volatile)
        ]
    else:
        x1 = [
            Variable(fact[0].cuda(), volatile=volatile),
            fact[1].cuda(),
            fact[2].cuda(),
            Variable(fact[3].cuda(), volatile=volatile)
        ]  # yapf: disable
    if not args.uncondition:
        if args.direction:
            x2 = [
                Variable(conjecture[0].cuda(), volatile=volatile),
                conjecture[1].cuda(),
                conjecture[2].cuda(),
                Variable(conjecture[3].cuda(), volatile=volatile),
                conjecture[4].cuda(),
                conjecture[5].cuda(),
                Variable(conjecture[6].cuda(), volatile=volatile)
            ]  # yapf: disable
        elif args.binary:
            x2 = [
                Variable(conjecture[0].cuda(), volatile=volatile), conjecture[1].cuda(),
                conjecture[2].cuda(), Variable(conjecture[3].cuda(), volatile=volatile),
                conjecture[4].cuda(), conjecture[5].cuda(), Variable(
                    conjecture[6].cuda(),
                    volatile=volatile), conj_bi[0].cuda(), Variable(
                        conj_bi[1].cuda(), volatile=volatile)
            ]
        else:
            x2 = [
                Variable(conjecture[0].cuda(), volatile=volatile),
                conjecture[1].cuda(),
                conjecture[2].cuda(),
                Variable(conjecture[3].cuda(),volatile=volatile)
            ]  # yapf: disable
    y = Variable(torch.Tensor([y]).long().cuda(), requires_grad=False)

    pair_total += x1[1].size()[0]
    if not args.uncondition:
        pair_total += x2[1].size()[0]

    if not args.uncondition:
        feat2 = net(x2)

    if args.add_conj:
        feat1 = net(x1, feat2[-1].max(0)[0])
    else:
        feat1 = net(x1)

    for k in range(len(feat1)):
        if args.loss == 'mulloss':
            catfeat[k].append((feat2[k].max(0)[0], feat1[k].max(0)[0]))
        elif args.loss == 'condloss':
            catfeat[k].append((feat2[k].max(0)[0], feat1[k].max(0)[0]))
        elif args.uncondition or args.add_conj:
            catfeat[k].append(feat1[k].max(0)[0])
        elif args.loss == 'concat':
            catfeat[k].append(model.maxpoolcat(feat1[k], feat2[k]))
        elif args.loss == 'concat_em_uc':
            catfeat[k].append(model.concat_em_uc(feat1[k], feat2[k]))
        elif args.loss == 'em':
            catfeat[k].append(model.em(feat1[k], feat2[k]))
        elif args.loss == 'mixmax':
            pair = (feat1[k].max(0)[0], model.dot_max(
                mid_net(feat1[k]), mid_net(feat2[k])))
            catfeat[k].append(pair)
        elif args.loss == 'mixmean':
            pair = (feat1[k].max(0)[0], model.dot_mean(
                mid_net(feat1[k]), mid_net(feat2[k])))
            catfeat[k].append(pair)
        elif args.loss == 'pair':
            cated = model.maxpoolcat(feat2[k], feat1[k])
            catfeat[k].append(net.pair_forward(cated))
        else:
            assert False, 'Wrong --loss option.'
    label.append(y)
    epoch_end = False
    return pair_total, epoch_end


def forward_step(net, mid_net, loss_fn, loader, args):
    pair_total = 0
    catfeat = []
    for i in range(args.loss_step):
        catfeat.append([])
    label = []
    epoch_end = False
    while pair_total < args.max_pair / args.worker:
        pair_total, epoch_end = embed_step(pair_total, net, mid_net, loss_fn, catfeat,
                                           label, loader, args, volatile=args.fix_net)
    losses = []
    for k in range(len(catfeat)):
        if args.fix_net:
            for feat in catfeat[k]:
                if isinstance(feat, tuple):
                    feat[0].volatile = False
                    feat[0].requires_grad = False
                    feat[1].volatile = False
                    feat[1].requires_grad = False
                else:
                    feat.volatile = False
                    feat.requires_grad = False
        losses.append(loss_fn[k](catfeat[k], label))
    loss = sum(losses)
    loss_total = loss.data[0]
    correct = []
    for k in range(len(loss_fn)):
        correct.append(loss_fn[k].check_result(label))
    total = len(label)
    net.zero_grad()
    if mid_net != None:
        mid_net.zero_grad()
    for l in loss_fn:
        l.zero_grad()
    loss.backward()
    return loss_total, correct, total, epoch_end


def test_forward(net, mid_net, loss_fn, loader, args):
    #for func in loss_fn:
    #    func.eval()
    # Variables for testing
    pair_total = 0
    catfeat = []
    label = []
    for i in range(args.loss_step):
        catfeat.append([])
    correct = 0
    total = 0
    epoch_end = False
    while not epoch_end:
        pair_total, epoch_end = embed_step(
            pair_total,
            net,
            mid_net,
            loss_fn,
            catfeat,
            label,
            loader,
            args,
            volatile=True)

        if pair_total > args.max_pair / args.worker or (pair_total != 0 and epoch_end):
            losses = []
            for k in range(len(loss_fn)):
                losses.append(loss_fn[k](catfeat[k], label))
            if not args.voting:
                correct += loss_fn[-1].check_result(label)
            else:
                scores = [loss.score for loss in loss_fn]
                score = sum(scores) / len(scores)
                score = (score - 0.5).sign() / 2 + 0.5
                y = torch.cat(label).data
                correct += score.eq(y).cpu().sum()
            total += len(label)
            # print(correct / total)
            pair_total = 0
            for k in range(len(catfeat)):
                catfeat[k] = []
            label = []
    return correct, total


def train(inqueues, outqueues, net, mid_net, loss_fn, opt, loaders, args, _logger):
    def _update_grad(net, mid_net, loss_fn, grad_list):
        if not args.fix_net:
            for name, param in net.named_parameters():
                grad_step = sum([data['total'] * data['net'][name] for data in grad_list
                                 ]) / sum([data['total'] for data in grad_list])
                param.grad = Variable(grad_step)

            if mid_net is not None:
                for name, param in mid_net.named_parameters():
                    grad_step = sum([
                        data['total'] * data['mid_net'][name] for data in grad_list
                    ]) / sum([data['total'] for data in grad_list])
                    param.grad = Variable(grad_step)

        for i in range(args.loss_step):
            for name, param in loss_fn[i].named_parameters():
                grad_step = sum([
                    data['total'] * data['loss_fn'][i][name] for data in grad_list
                ]) / sum([data['total'] for data in grad_list])
                param.grad = Variable(grad_step)

            for name in loss_fn[i].buffers.keys():
                loss_fn[i].buffers[name].copy_(grad_list[0]['buffer']['loss_fn'][i][name])

    def _update_correct(grad_list, corrects):
        for i in range(len(corrects)):
            corrects[i] += sum([data['correct'][i] for data in grad_list])

    def _step_stat(grad_list):
        step_sample_total = sum([data['total'] for data in grad_list])
        step_loss_total = sum([data['loss_total'] for data in grad_list])
        return step_sample_total, step_loss_total

    # Variables for training
    t = time.time()
    cur_epoch_training_total = 0
    training_total = 0
    valid_total = 0
    sample_total = 0
    loss_total = 0
    correct_total = [0] * (args.loss_step)

    recorder = None
    if args.record is not None:
        recorder = HistoryRecorder(args.record)

    epoch = 0
    epoch_end = False
    while epoch < args.epoch:
        data = {}

        if not args.fix_net or training_total == 0:
            data['fix_net'] = False
        else:
            data['fix_net'] = True

        data['args'] = args
        if not data['fix_net']:
            data['net'] = net.state_dict()
            if mid_net is not None:
                data['mid_net'] = mid_net.state_dict()

        data['loss_fn'] = []
        for loss in loss_fn:
            data['loss_fn'].append(loss.state_dict())
        data['test'] = False

        for i in range(args.worker):
            inqueues[i].send(data)

        grad_list = []
        for i in range(args.worker):
            data = outqueues[i].recv()
            grad_list.append(data)

        _update_grad(net, mid_net, loss_fn, grad_list)
        _update_correct(grad_list, correct_total)

        step_sample_total, step_loss_total = _step_stat(grad_list)

        cur_epoch_training_total += step_sample_total
        training_total += step_sample_total
        valid_total += step_sample_total
        sample_total += step_sample_total
        loss_total += step_loss_total
        opt.step()
        if (epoch + 1) * args.epoch_len <= training_total:
            _logger.info('Epoch END!!!')
            epoch_end = True

        if sample_total > args.observe or epoch_end:
            end_str = ' END!' if epoch_end else ''
            _logger.info('Epoch: %d%s Iteration: %d Loss: %.5f perTime: %.3f', epoch,
                         end_str, cur_epoch_training_total, loss_total / sample_total,
                         (time.time() - t) / sample_total)
            accs = []
            for k in range(len(loss_fn)):
                accs.append('acc %d: %.5f' % (k, correct_total[k] / sample_total))
            if recorder is not None:
                recorder.train_acc(training_total, correct_total[-1] / sample_total)
                recorder.save_record()
            _logger.info(' '.join(accs))
            sample_total = 0
            loss_total = 0
            correct_total = [0] * (args.loss_step)
            t = time.time()

        if valid_total > args.check_num or (epoch_end and epoch == args.epoch - 1):
            aux = {}
            aux['epoch'] = epoch
            aux['cur_iter'] = cur_epoch_training_total
            aux['total_iter'] = training_total
            save_model(aux, args, net, mid_net, loss_fn,
                       args.output + '_%d_%d' % (epoch, cur_epoch_training_total))
            _logger.warning('Model saved to %s',
                            args.output + '_%d_%d' % (epoch, cur_epoch_training_total))
            _logger.warning('Start validation!')
            valid_start = time.time()
            data = {}
            data['fix_net'] = False
            data['args'] = args
            data['net'] = net.state_dict()
            if mid_net is not None:
                data['mid_net'] = mid_net.state_dict()
            data['loss_fn'] = []
            for loss in loss_fn:
                data['loss_fn'].append(loss.state_dict())
            data['test'] = True

            for i in range(args.worker):
                inqueues[i].send(data)

            result_correct = 0
            result_total = 0

            for i in range(args.worker):
                data = outqueues[i].recv()
                result_correct += data['correct']
                result_total += data['total']
            result_ = result_correct / result_total
            _logger.warning('Validation complete! Time lapse: %.3f, Test acc: %.5f' %
                            (time.time() - valid_start, result_))

            if recorder is not None:
                recorder.test_acc(training_total, result_)
                recorder.save_record()
            valid_total = 0
            if args.fix_net:
                _logger.warning('learning rate decreases from %.6f to %.6f',
                                args.learning_rate, args.learning_rate / 3)
                args.learning_rate /= 2
                opt = get_opt(net, mid_net, loss_fn, args)

        if args.unfix_net_after is not None and training_total > args.unfix_net_after:
            args.fix_net = False

        if epoch_end and args.learning_rate > args.min_lr:
            _logger.warning('learning rate decreases from %.6f to %.6f',
                            args.learning_rate, args.learning_rate / 3)
            args.learning_rate /= 3
            opt = get_opt(net, mid_net, loss_fn, args)

        if epoch_end:
            cur_epoch_training_total = 0
            epoch_end = False
            epoch += 1


def worker(inqueue, outqueue, loader, args, worker_no):
    def _grad_dict(net):
        if net is None:
            return None
        else:
            return {name: param.grad.data for name, param in net.named_parameters()}

    _logger = log.get_logger('worker%d' % worker_no, args, append=True)

    net, mid_net, loss_fn = create_models(args, loader)
    _logger.info('Here I am')
    while True:
        data = inqueue.recv()
        args = data['args']
        fix_net = data['fix_net']
        if not fix_net:
            net.load_state_dict(data['net'])
            if mid_net is not None:
                mid_net.load_state_dict(data['mid_net'])
        for i in range(len(loss_fn)):
            loss_fn[i].load_state_dict(data['loss_fn'][i])
        test = data['test']
        if test is True:
            for loss in loss_fn:
                loss.eval()
            test_loader = data_loader.DataLoader(
                args.test_file,
                args.dict_file,
                separate_conj_stmt=args.direction,
                binary=args.binary,
                part_no=worker_no,
                part_total=args.worker,
                norename=args.norename,
                filter_abelian=args.fabelian,
                compatible=args.compatible)
            test_loader.start_reader()
            correct, total = test_forward(net, mid_net, loss_fn, test_loader, args)
            test_loader.destruct()
            data = {}
            data['correct'] = correct
            data['total'] = total
            for loss in loss_fn:
                loss.train()
        else:  # train
            loss_total, correct, total, epoch_end = forward_step(
                net, mid_net, loss_fn, loader, args)
            if epoch_end:
                _logger.warning('My current epoch ends!')
            data = {}
            if not args.fix_net:
                data['net'] = _grad_dict(net)
                data['mid_net'] = _grad_dict(mid_net)
            data['loss_fn'] = []
            for loss in loss_fn:
                data['loss_fn'].append(_grad_dict(loss))
            data['total'] = total
            data['loss_total'] = loss_total
            data['correct'] = correct
            data['buffer'] = {}
            data['buffer']['loss_fn'] = [{} for _ in loss_fn]
            for i in range(len(loss_fn)):
                for name, b in named_buffers(loss_fn[i]):
                    data['buffer']['loss_fn'][i][name] = b
        # Send data!
        outqueue.send(data)


if __name__ == '__main__':
    main()
