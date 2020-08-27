import os
import sys
import pickle
from numpy.lib.format import open_memmap
from ntu_read_skeleton import read_xyz


#################################################
origin_path = '../ntu-rgb+d-skeletons120'
output_dir_path = 'Data/NTU-RGB+D120'
ignore_sample_path = 'Tools/Gen_dataset/ignore_sample_ntu120.txt'
#################################################

# cross-subject
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
                     38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
                     70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
# cross-setup
training_setup = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# config
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 50


def make_data_ntu(origin_path, output_path, benchmark, part):
    if ignore_sample_path != None:
        with open(ignore_sample_path, 'r') as f:
            ignored_sample = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_sample = []

    sample_name = []
    sample_label = []

    for filename in os.listdir(origin_path):
        if filename in ignored_sample:
            continue

        setup_id = int(filename[filename.find('S') + 1:filename.find('S') + 4])
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])

        if benchmark == 'xsetup':
            istraining = (setup_id in training_setup)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'test':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class-1)

    with open('{}/{}_label.pkl'.format(output_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

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
    benchmark = ['xsub', 'xsetup']
    part = ['train', 'test']

    for b in benchmark:
        for p in part:
            output_path = os.path.join(output_dir_path, b)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            make_data_ntu(origin_path, output_path, benchmark=b, part=p)
