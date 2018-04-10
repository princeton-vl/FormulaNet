import os
import argparse
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe

import log
from model_utils import load_model
from model_utils import print_args
from batch_train import worker


def main():
    parser = argparse.ArgumentParser(description='The ultimate tester')
    parser.add_argument(
        '--model', type=str, help='The model file name used for testing', required=True)
    parser.add_argument(
        '--model_path', type=str, help='The path to model folder', default='../models')
    parser.add_argument('--log', type=str, help='Path to log file', required=True)
    parser.add_argument(
        '--data',
        type=str,
        help='Path to testing set folder',
        default='../data/hol_data/test')
    parser.add_argument('--worker', type=int, help='Number of workers', default=4)
    parser.add_argument('--max_pair', type=int, help='Change max_pair settings')
    parser.add_argument('--compatible', action='store_true', help='Use compatible mode to run.')
    parser.add_argument('--dict_file', type=str, help='Replace dict')

    settings = parser.parse_args()
    mp.set_start_method('spawn')  # Using spawn is decided.

    _logger = log.get_logger(__name__, settings)
    _logger.info('Test program parameters')
    _logger.info(print_args(settings))
    model_path = os.path.join(settings.model_path, settings.model)
    net, mid_net, loss_fn, test_loader, args = load_model(model_path, settings.data, settings.compatible)
    args.test_file = settings.data
    if settings.dict_file is not None:
        args.dict_file = settings.dict_file
    if settings.max_pair is not None:
        args.max_pair = settings.max_pair
    args.compatible = settings.compatible 
    inqueues = []
    outqueues = []

    if settings.worker is not None:
        args.worker = settings.worker

    plist = []
    for i in range(args.worker):
        recv_p, send_p = Pipe(False)
        recv_p2, send_p2 = Pipe(False)
        inqueues.append(send_p)
        outqueues.append(recv_p2)
        plist.append(
            Process(target=worker, args=(recv_p, send_p2, test_loader, args, i)))
        plist[-1].start()

    test_loader = None

    _logger.info('Model parameters')
    _logger.info(print_args(args))

    valid_start = time.time()
    data = {}
    data['args'] = args
    data['net'] = net.state_dict()
    data['fix_net'] = False
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

    _logger.warning('Test start!')
    for i in range(args.worker):
        data = outqueues[i].recv()
        result_correct += data['correct']
        result_total += data['total']
    result_ = result_correct / result_total
    _logger.warning('Validation complete! Time lapse: %.3f, Test acc: %.5f' %
                    (time.time() - valid_start, result_))

    for p in plist:
        p.terminate()
    time.sleep(5)

if __name__ == '__main__':
    main()
