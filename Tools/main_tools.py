import datetime
from collections import OrderedDict
import torch

from Tools.Model.sta_gcn import STA_GCN as STA_GCN


def load_model(model_args, log_dir, device, output_device):
    model_name = model_args.pop('model_name')
    if model_name == 'STA_GCN':
        model_class = STA_GCN
    else:
        raise ValueError

    model = model_class(num_class=model_args['num_class'],
                        in_channels=model_args['in_channels'],
                        residual=model_args['residual'],
                        dropout=model_args['dropout'],
                        num_person=model_args['num_person'],
                        t_kernel_size=model_args['t_kernel_size'],
                        layout=model_args['layout'],
                        strategy=model_args['strategy'],
                        hop_size=model_args['hop_size'],
                        num_att_A=model_args['num_att_A'])

    model = model.to(output_device)

    # load weight
    if model_args['load_weight']:
        print('# --- LOAD WEIGHT!!! ---')
        weights_path = log_dir + '/best_model.pt'
        print('# Load weights from: ', weights_path)
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1], v.to(output_device)] for k, v in weights.items()])
        try:
            model.load_state_dict(weights)
        except:
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print('# Can not find these weights:')
            for d in diff:
                print('# ' + d)
            state.update(weights)
            model.load_state_dict(state)

    # Multi GPU
    if type(device) is list:
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device, output_device=output_device)

    return model

def load_optimizer(optimizer_args, model, momentum=0.9, nesterov=True, weight_decay=0.0001):
    if optimizer_args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=optimizer_args['base_lr'],
                                    momentum=momentum,
                                    nesterov=nesterov,
                                    weight_decay=weight_decay)
    elif optimizer_args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=optimizer_args['base_lr'],
                                     weight_decay=weight_decay)
    else:
        raise ValueError

    return optimizer

def print_log(input, log_dir, print_tf=True, log_tf=True):
    if print_tf:
        print(input)
    if log_tf:
        with open('{}/log.txt'.format(log_dir), 'a') as f:
            f.write(input+'\n')

def header_print(log_dir):
    print_log(
        '# -------------------------------------------------------------------------------------------------------------------------------\n'
        '# | epoch |:|train|:| mean loss | accuracy[AB] | accuracy[PB] |:|test|:| mean loss | accuracy[AB] | accuracy[PB] |:| current- |:|\n'
        '# -------------------------------------------------------------------------------------------------------------------------------',
        log_dir)

def result_print(epoch, log_dir, train_mean_loss, train_accuracy_ab, train_accuracy_pb, test_mean_loss,
                 test_accuracy_ab, test_accuracy_pb):
    # Print  [9 = len(mean loss), 8 = len(accuracy)]
    train_mean_loss = str(round(train_mean_loss, 4)).rjust(9, ' ')
    train_accuracy_ab = str(round(train_accuracy_ab, 3)).rjust(12, ' ')
    train_accuracy_pb = str(round(train_accuracy_pb, 3)).rjust(12, ' ')
    test_mean_loss = str(round(test_mean_loss, 4)).rjust(9, ' ')
    test_accuracy_ab = str(round(test_accuracy_ab, 3)).rjust(12, ' ')
    test_accuracy_pb = str(round(test_accuracy_pb, 3)).rjust(12, ' ')

    print_log('# | {:5d} |:|     |:| {} | {} | {} |:|    |:| {} | {} | {} |:| {} |:|'.format(
        epoch, train_mean_loss, train_accuracy_ab, train_accuracy_pb, test_mean_loss, test_accuracy_ab,
        test_accuracy_pb, datetime.datetime.now().strftime('%H:%M:%S')), log_dir)
