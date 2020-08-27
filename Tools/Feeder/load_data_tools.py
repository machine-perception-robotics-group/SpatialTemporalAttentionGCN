import torch
from .feeder import Feeder as Feeder


def load_data(feeder_args, train_feeder_args, test_feeder_args):
    frame_size = feeder_args.pop('frame_size')
    batch_size = feeder_args.pop('batch_size')
    num_worker = feeder_args.pop('num_worker')

    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(data_path=train_feeder_args['data_path'],
                                                                      label_path=train_feeder_args['label_path'],
                                                                      frame_size=frame_size,
                                                                      normalization=train_feeder_args['normalization'],
                                                                      random_shift=train_feeder_args['random_shift'],
                                                                      valid_choose=train_feeder_args['valid_choose'],
                                                                      frame_thinning=train_feeder_args['frame_thinning'],
                                                                      random_choose=train_feeder_args['random_choose'],
                                                                      repeat_padding=train_feeder_args['repeat_padding'],
                                                                      random_move=train_feeder_args['random_move'],
                                                                      add_noise=train_feeder_args['add_noise'],
                                                                      frame_normalization=train_feeder_args['frame_normalization']),
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_worker,
                                                       pin_memory=True)
    data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(data_path=test_feeder_args['data_path'],
                                                                     label_path=test_feeder_args['label_path'],
                                                                     frame_size=frame_size,
                                                                     normalization=test_feeder_args['normalization'],
                                                                     random_shift=test_feeder_args['random_shift'],
                                                                     valid_choose=test_feeder_args['valid_choose'],
                                                                     frame_thinning=test_feeder_args['frame_thinning'],
                                                                     random_choose=test_feeder_args['random_choose'],
                                                                     repeat_padding=test_feeder_args['repeat_padding'],
                                                                     random_move=test_feeder_args['random_move'],
                                                                     add_noise=test_feeder_args['add_noise'],
                                                                     frame_normalization=test_feeder_args['frame_normalization']),
                                                      batch_size=batch_size * 2,
                                                      shuffle=False,
                                                      num_workers=0)

    return data_loader
