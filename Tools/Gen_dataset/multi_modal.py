import argparse
import numpy as np
from tqdm import tqdm
import torch


def gen_multi_modal(data_path, output_dir_path, bone, part, dataset, frame_size=300):
    joint_data = np.load(data_path)
    joint_data = torch.from_numpy(joint_data)

    ### coordinate data
    print('# make coordinate data. . .')
    origin_data = joint_data.clone()
    bone_data = joint_data.clone()
    # core
    for v1, v2 in bone:
        if dataset != 'kinetics-skeleton':
            v1 -= 1
            v2 -= 1
        bone_data[:, :, :, v1, :] = origin_data[:, :, :, v1, :] - origin_data[:, :, :, v2, :]
    coordinate_data = torch.cat([joint_data, bone_data], dim = 1)
    # unleash the captive memory
    del joint_data, origin_data, bone_data
    # save
    print('# save...')
    coordinate_data = coordinate_data.numpy()
    output_path = output_dir_path + part + '_coordinate.npy'
    np.save(output_path, coordinate_data)

    ### velocity data
    print('# make velocity data. . .')
    coordinate_data = torch.from_numpy(coordinate_data)
    velocity_data = coordinate_data.clone()
    # core
    for t in tqdm(range(frame_size-1, 0, -1), desc='# generating'):
        velocity_data[:, :, t, :, :] = coordinate_data[:, :, t - 1, :, :] - coordinate_data[:, :, t, :, :]
    velocity_data[:, :, 0, :, :] = coordinate_data[:, :, 0, :, :] - coordinate_data[:, :, 0, :, :]
    # unleash the captive memory
    del coordinate_data
    # save
    print('# save...')
    velocity_data = velocity_data.numpy()
    output_path = output_dir_path + part + '_velocity.npy'
    np.save(output_path, velocity_data)

    ### acceleration data
    print('# make acceleration data. . .')
    velocity_data = torch.from_numpy(velocity_data)
    acceleration_data = velocity_data.clone()
    # core
    for t in tqdm(range(1, frame_size), desc='# generating'):
        acceleration_data[:, :, t, :, :] = ((velocity_data[:, :, t, :, :] ** 2) - (velocity_data[:, :, t - 1, :, :] ** 2)) / 2
    acceleration_data[:, :, 0, :, :] = velocity_data[:, :, 0, :, :] - velocity_data[:, :, 0, :, :]
    # save
    print('# save...')
    acceleration_data = acceleration_data.numpy()
    output_path = output_dir_path + part + '_acceleration.npy'
    np.save(output_path, acceleration_data)
    # unleash the captive memory
    del acceleration_data

def ntu60():
    benchmark = ['xsub', 'xview']
    part = ['train', 'test']
    bone = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))

    for b in benchmark:
        for p in part:
            print('# {}_{}'.format(b, p))
            data_path = ('Data/NTU-RGB+D60/{}/{}_joint.npy'.format(b, p))
            output_dir_path = ('Data/NTU-RGB+D60/{}/'.format(b, p))
            gen_multi_modal(data_path, output_dir_path, bone, p, dataset='ntu60')

def ntu120():
    benchmark = ['xsub', 'xsetup']
    part = ['train', 'test']
    bone = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))

    for b in benchmark:
        for p in part:
            print('# {}_{}'.format(b, p))
            data_path = ('Data/NTU-RGB+D120/{}/{}_joint.npy'.format(b, p))
            output_dir_path = ('Data/NTU-RGB+D120/{}/'.format(b, p))
            gen_multi_modal(data_path, output_dir_path, bone, p, dataset='ntu120')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi_modal_config')
    parser.add_argument('--dataset', type=str, required=True, choices=['ntu60', 'ntu120'])
    p = parser.parse_args()

    if p.dataset == 'ntu60':
        print('# gen multi_modal: NTU-RGB+D60')
        ntu60()

    elif p.dataset == 'ntu120':
        print('# gen multi_modal: NTU-RGB+D120')
        ntu120()
