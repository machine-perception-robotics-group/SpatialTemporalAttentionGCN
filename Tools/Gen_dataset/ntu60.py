import os
import sys
import pickle
from numpy.lib.format import open_memmap
from ntu_read_skeleton import read_xyz


#################################################
origin_path = '../ntu-rgb+d-skeletons60'
output_dir_path = 'Data/NTU-RGB+D60'
ignore_sample_path = 'Tools/Gen_dataset/ignore_sample_ntu60.txt'
#################################################

# cross-subject
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# cross-view
training_cameras = [2, 3]

max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 50


def make_data_ntu(origin_path, output_path, benchmark, part):
    # ignore sample
    if ignore_sample_path != None:
        with open(ignore_sample_path, 'r') as f:
            ignored_sample = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_sample = []

    sample_name = []
    sample_label = []

    # start
    for filename in os.listdir(origin_path):
        # pass: ignore sample
        if filename in ignored_sample:
            continue

        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        # xsub or xview
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        # train or test
        if part == 'train':
            issample = istraining
        elif part == 'test':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            # storage: data name
            sample_name.append(filename)
            # storage: label
            sample_label.append(action_class-1)

    # store: label
    with open('{}/{}_label.pkl'.format(output_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    # store: data
    fp = open_memmap('{}/{}_joint.npy'.format(output_path, part),
                     dtype='float32',
                     mode='w+',
                     shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Writing {:>5}-{:<5} data: '.format(i + 1, len(sample_name), benchmark, part))
        data = read_xyz(os.path.join(origin_path, s),
                        max_body=max_body,
                        num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


def print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")


if __name__ == '__main__':
    benchmark = ['xsub', 'xview']
    part = ['train', 'test']

    for b in benchmark:
        for p in part:
            output_path = os.path.join(output_dir_path, b)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            make_data_ntu(origin_path, output_path, benchmark=b, part=p)
