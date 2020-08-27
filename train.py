import os
import csv
import yaml
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch

import Tools.main_tools as main_tools
import Tools.Feeder.load_data_tools as load_data_tools


class Processor:
    def __init__(self, args, config):
        log_dir = args.pop('log_dir')
        print('\n# log_dir: ', log_dir)

        # mkdir log_dir
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            with open(log_dir + '/log.txt', 'w'):
                pass
        else:
            print('# warning: log_dir[{}] already exist.'.format(log_dir))
            answer = input('# continue? y/n : ')
            if answer == 'y':
                pass
            else:
                print('# bye :)')
                exit()

        # save config
        main_tools.print_log('# Parameters:\n{}\n'.format(str(args)), log_dir=log_dir, print_tf=False)
        shutil.copyfile(config, os.path.join(log_dir, 'config.yaml'))

        # save Model dir
        model_path = os.path.join(log_dir, 'Model')
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.copytree('./Tools/Model', model_path)

        # device
        device = args.pop('device')
        self.output_device = device[0] if type(device) is list else device

        # model
        model_args = args.pop('model_args')
        print('# Load model')
        print('#    model_name   : ', model_args['model_name'])
        print('#    num_class    : ', model_args['num_class'])
        print('#    in_channels  : ', model_args['in_channels'])
        print('#    residual     : ', model_args['residual'])
        print('#    dropout      : ', model_args['dropout'])
        print('#    num_person   : ', model_args['num_person'])
        print('#    t_kernel_size: ', model_args['t_kernel_size'])
        print('# graph args')
        print('#    layout   : ', model_args['layout'])
        print('#    strategy : ', model_args['strategy'])
        print('#    hop_size : ', model_args['hop_size'])
        print('#    num_att_A: ', model_args['num_att_A'])
        self.model = main_tools.load_model(model_args, log_dir, device, self.output_device)

        # optimizer
        optimizer_args = args.pop('optimizer_args')
        self.base_lr = optimizer_args['base_lr']
        self.lr_step = optimizer_args['lr_step']
        print('# Load optimizer')
        print('# optimizer: ', optimizer_args['optimizer'])
        print('# base_lr  : ', self.base_lr)
        print('# lr_step  : ', self.lr_step)
        self.optimizer = main_tools.load_optimizer(optimizer_args, self.model)

        # loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # data
        feeder_args = args.pop('feeder_args')
        train_feeder_args = feeder_args.pop('train_feeder_args')
        test_feeder_args = feeder_args.pop('test_feeder_args')
        print('# load data')
        print('#    train_data : ', train_feeder_args['data_path'])
        print('#    train_label: ', train_feeder_args['label_path'])
        print('#    test_data  : ', test_feeder_args['data_path'])
        print('#    test_label : ', test_feeder_args['label_path'])
        print('#    frame size : ', feeder_args['frame_size'])
        print('#    batch size : ', feeder_args['batch_size'])
        print('#    num worker : ', feeder_args['num_worker'])
        print('# data processing       : train/test')
        print('#    normalization      : {} / {}'.format(train_feeder_args['normalization'], test_feeder_args['normalization']))
        print('#    random_shift       : {} / {}'.format(train_feeder_args['random_shift'], test_feeder_args['random_shift']))
        print('#    valid_choose       : {} / {}'.format(train_feeder_args['valid_choose'], test_feeder_args['valid_choose']))
        print('#    frame_thinning     : {} / {}'.format(train_feeder_args['frame_thinning'], test_feeder_args['frame_thinning']))
        print('#    random_choose      : {} / {}'.format(train_feeder_args['random_choose'], test_feeder_args['random_choose']))
        print('#    repeat_padding     : {} / {}'.format(train_feeder_args['repeat_padding'], test_feeder_args['repeat_padding']))
        print('#    random_move        : {} / {}'.format(train_feeder_args['random_move'], test_feeder_args['random_move']))
        print('#    add_noise          : {} / {}'.format(train_feeder_args['add_noise'], test_feeder_args['add_noise']))
        print('#    frame_normalization: {} / {}'.format(train_feeder_args['frame_normalization'], test_feeder_args['frame_normalization']))
        self.data_loader = load_data_tools.load_data(feeder_args, train_feeder_args, test_feeder_args)

        # log list
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []

        start_epoch = args.pop('start_epoch')
        end_epoch = args.pop('end_epoch')

        # magic spell
        if args['cudnn_benchmark']:
            torch.backends.cudnn.benchmark = True

        print('# start epoch    : ', start_epoch)
        print('# end epoch      : ', end_epoch)
        print('# device         : ', device)
        print('# cudnn_benchmark: ', args['cudnn_benchmark'])

        # start
        print('\n# Start! :)')
        for e in range(start_epoch, end_epoch+1):
            if ((e - 1) % 10) == 0:
                main_tools.header_print(log_dir)

            self.adjust_learning_rate(e)

            train_mean_loss, train_accuracy_ab, train_accuracy_pb = self.train()
            test_mean_loss, test_accuracy_ab, test_accuracy_pb = self.test()

            self.train_loss_list.append(train_mean_loss)
            self.train_acc_list.append(train_accuracy_pb)
            self.test_loss_list.append(test_mean_loss)
            self.test_acc_list.append(test_accuracy_pb)
            main_tools.result_print(e, log_dir, train_mean_loss, train_accuracy_ab, train_accuracy_pb,
                                                test_mean_loss, test_accuracy_ab, test_accuracy_pb)

            # save best model
            if (len(self.test_acc_list) - 1) == np.argmax(self.test_acc_list):
                path = log_dir + '/best_model.pt'
                torch.save(self.model.state_dict(), path)

        main_tools.print_log('# Best test accuracy: {:.3f}[%]'.format(np.max(self.test_acc_list)), log_dir)

        # plot csv
        self.train_loss_list.insert(0, 'train_loss')
        self.train_acc_list.insert(0, 'train_acc')
        self.test_loss_list.insert(0, 'test_loss')
        self.test_acc_list.insert(0, 'test_acc')
        with open(log_dir + '/loss_accuracy.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(self.train_loss_list)
            writer.writerow(self.train_acc_list)
            writer.writerow(self.test_loss_list)
            writer.writerow(self.test_acc_list)

        print('\n# bye :)\n')

    def train(self):
        loss_list = []
        correct_ab = 0
        correct_pb = 0
        for batch_idx, (data, label, _) in enumerate(tqdm(self.data_loader['train'], leave=False, desc='# train')):
            data = data.requires_grad_().to(self.output_device)
            label = label.to(self.output_device)

            output_ab, output_pb, _, _ = self.model(data)

            loss = self.loss(output_ab, label) + self.loss(output_pb, label)
            loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predict = torch.max(output_ab.data, 1)
            correct_ab += (predict == label).sum().item()
            _, predict = torch.max(output_pb.data, 1)
            correct_pb += (predict == label).sum().item()

        len_data_loader = len(self.data_loader['train'].dataset)
        return np.mean(loss_list), (100. * correct_ab/len_data_loader), (100. * correct_pb/len_data_loader)

    def test(self):
        loss_list = []
        correct_ab = 0
        correct_pb = 0
        with torch.no_grad():
            for batch_idx, (data, label, _) in enumerate(tqdm(self.data_loader['test'], leave=False, desc='#  test')):
                data = data.to(self.output_device)
                label = label.to(self.output_device)

                output_ab, output_pb, _, _ = self.model(data)

                loss = self.loss(output_ab, label) + self.loss(output_pb, label)
                loss_list.append(loss.item())

                _, predict = torch.max(output_ab.data, 1)
                correct_ab += (predict == label).sum().item()
                _, predict = torch.max(output_pb.data, 1)
                correct_pb += (predict == label).sum().item()

        len_data_loader = len(self.data_loader['test'].dataset)
        return np.mean(loss_list), (100. * correct_ab/len_data_loader), (100. * correct_pb/len_data_loader)

    def adjust_learning_rate(self, epoch):
        lr = self.base_lr * (0.1 ** np.sum(epoch >= np.array(self.lr_step)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def init_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main config')
    parser.add_argument('--config', type=str, default=None)
    p = parser.parse_args()

    init_seed()

    # load config
    with open(p.config, 'r') as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)

    # start Processor
    Processor(args, p.config)