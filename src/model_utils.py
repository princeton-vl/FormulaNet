import os
import torch
from torch.autograd import Variable
from collections import OrderedDict

import model
import data_loader
import loss


class FakeVariable(Variable):
    data = None  # Used to override some tricks in pytorch
    grad = None  # Used to override some tricks in pytorch

    def __init__(self, var):
        self.data = var  #.share_memory_()
        self.grad = None
        self.requires_grad = True


class FakeModule:
    def __init__(self, module):
        params, buffers = FakeModule.init_state_dict(module)
        self.params = params
        self.buffers = buffers

    @staticmethod
    def init_state_dict(module, params=None, buffers=None, prefix=''):
        if params is None:
            params = OrderedDict()
        if buffers is None:
            buffers = OrderedDict()
        for name, param in module._parameters.items():
            if param is not None:
                params[prefix + name] = FakeVariable(param.data)
        for name, buf in module._buffers.items():
            if buf is not None:
                buffers[prefix + name] = buf
        for name, smodule in module._modules.items():
            if smodule is not None:
                FakeModule.init_state_dict(smodule, params, buffers,
                                           prefix + name + '.')
        return params, buffers

    def state_dict(self):
        params = {name: param.data for name, param in self.params.items()}
        buffers = {name: buf for name, buf in self.buffers.items()}
        params.update(buffers)
        return params

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
            if isinstance(param, FakeVariable):
                param = param.data
            own_state[name].copy_(param)

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def parameters(self):
        for param in self.params.values():
            yield param

    def named_parameters(self):
        for name, param in self.params.items():
            yield name, param

    def named_buffers(self):
        for name, buf in self.buffers.items():
            yield name, buf


class HistoryRecorder:
    def __init__(self, record_file):
        self.file = record_file
        self.test_history = []
        self.train_history = []

    def train_acc(self, training_total, acc):
        self.train_history.append((training_total, acc))

    def test_acc(self, training_total, acc):
        self.test_history.append((training_total, acc))

    def save_record(self):
        dict_file = {}
        dict_file['train'] = self.train_history
        dict_file['test'] = self.test_history
        torch.save(dict_file, self.file)


def save_model(aux, args, net, mid_net, loss_fn, out_path):
    savedata = {}
    savedata['aux'] = aux
    savedata['args'] = args
    savedata['net'] = {'state_dict': net.state_dict()}
    if mid_net is not None:
        savedata['mid_net'] = {'state_dict': mid_net.state_dict()}
    savedata['loss_fn'] = []
    for fn in loss_fn:
        savedata['loss_fn'].append({'state_dict': fn.state_dict()})
    torch.save(savedata, out_path)


def load_model(model_file, overided_path=None, compatible=False, subdir=None, noloader=False):
    data = torch.load(model_file)
    args = data['args']

    if not noloader:
        if overided_path is not None:
            if subdir is not None:
                data_path = os.path.join(overided_path, subdir)
            else:
                data_path = overided_path
        else:
            if subdir is not None:
                data_path = os.path.join(args.data_path, subdir)
            else:
                data_path = args.train_file
    if not hasattr(args, 'direction'):
        setattr(args, 'direction', args.separate_in_out)
    if not hasattr(args, 'loss'):
        setattr(args, 'loss', 'concat')
    if not hasattr(args, 'step2'):
        setattr(args, 'step2', False)
    if not hasattr(args, 'binary'):
        setattr(args, 'binary', False)
    if not hasattr(args, 'norename'):
        setattr(args, 'norename', False)
    if not hasattr(args, 'fabelian'):
        setattr(args, 'fabelian', False)
    if not hasattr(args, 'dropout'):
        setattr(args, 'dropout', 0)
    if not hasattr(args, 'no_step_supervision'):
        setattr(args, 'no_step_supervision', False)
    if not hasattr(args, 'cond_short_cut'):
        setattr(args, 'cond_short_cut', False)
    if not hasattr(args, 'loss_step'):
        setattr(args, 'loss_step', args.nSteps + 1)
    if not hasattr(args, 'loss_layers'):
        setattr(args, 'loss_layers', None)
    if not hasattr(args, 'tied_weight'):
        setattr(args, 'tied_weight', None)
    if not hasattr(args, 'resume'):
        setattr(args, 'resume', False)
    if not hasattr(args, 'fix_net'):
        setattr(args, 'fix_net', False)
    if not hasattr(args, 'compatible'):
        setattr(args, 'compatible', False)
    if not hasattr(args, 'resume_only_net'):
        setattr(args, 'resume_only_net', False)
    if not hasattr(args, 'unfix_net_after'):
        setattr(args, 'unfix_net_after', None)

    loader = None
    if not noloader:
        loader = data_loader.DataLoader(
            data_path, args.dict_file, separate_conj_stmt=args.direction, binary=args.binary)
        loader.iter_ = data['aux']['epoch'] + 1
        loader.total_iter = data['aux']['total_iter']

    net = model.GraphNet(loader.dict_size, args.nFeats, args.nSteps, args.block,
                         args.module_depth, args.bias, args.short_cut, args.direction,
                         args.loss, args.binary,
                         no_step_supervision=args.no_step_supervision,
                         compatible=compatible).cuda()

    net.load_state_dict(data['net']['state_dict'])
    net.train()
    mid_net = None
    if args.loss in ('mixmax', 'mixmean'):
        mid_net = model.FullyConnectedNet(
            args.nFeats, args.nFeats // 2, bias=args.bias).cuda()
        mid_net.load_state_dict(data['mid_net']['state_dict'])
        mid_net.train()

    loss_fn = []
    for i in range(args.loss_step):
        if args.loss == 'condloss':
            loss_fn.append(loss.CondLoss(args.nFeats * 2, args.nFeats,
                                         layer_list=args.loss_layers,
                                         dropout=args.dropout,
                                         cond_short_cut=args.cond_short_cut).cuda())
        elif args.loss in ('concat', 'concat_em_uc'):
            if args.uncondition or args.add_conj:
                loss_fn.append(loss.ClassifyLoss(args.nFeats, args.nFeats // 2,
                                                 layer_list=args.loss_layers,
                                                 dropout=args.dropout,
                                                 bias=compatible).cuda())
            else:
                loss_fn.append(loss.ClassifyLoss(args.nFeats * 2, args.nFeats,
                                                 layer_list=args.loss_layers,
                                                 dropout=args.dropout,
                                                 bias=compatible).cuda())
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
        loss_fn[-1].load_state_dict(data['loss_fn'][i]['state_dict'])
        loss_fn[-1].eval()

    return net, mid_net, loss_fn, loader, args


def print_args(args):
    s = '\nParameter setting:\n'
    for name, value in args.__dict__.items():
        s += str(name) + ': ' + str(value) + '\n'
    return s


def get_opt(net, mid_net, loss_fn, args, mode='all'):
    opt = None
    params = []
    if mode in ('all', 'net'):
        params += list(net.parameters())
        if mid_net is not None:
            params += list(mid_net.parameters())

    if mode in ('all', 'loss'):
        for func in loss_fn:
            params += list(func.parameters())

    if args.optim == 'sgd':
        opt = torch.optim.SGD(
            params=params, lr=args.learning_rate, weight_decay=1e-04, momentum=0.9)
    elif args.optim == 'rmsprop':
        opt = torch.optim.RMSprop(
            params=params, lr=args.learning_rate, weight_decay=1e-04)
    elif args.optim == 'adam':
        opt = torch.optim.Adam(params=params, lr=args.learning_rate, weight_decay=1e-04)
    else:
        assert opt is None, 'Wrong optimizer argument.'
    return opt


def named_buffers(self, memo=None, prefix=''):
    '''Get named buffer in given module'''
    if memo is None:
        memo = set()
    for name, p in self._buffers.items():
        if p is not None and p not in memo:
            memo.add(p)
            yield prefix + ('.' if prefix else '') + name, p
    for mname, module in self.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in named_buffers(module, memo, submodule_prefix):
            yield name, p


def split_list(l, total, i):
    persplit = len(l) // total
    offset = len(l) % total
    if i < offset:
        return l[persplit * i + i:persplit * (i + 1) + i + 1]
    else:
        return l[persplit * i + offset:persplit * (i + 1) + offset]
