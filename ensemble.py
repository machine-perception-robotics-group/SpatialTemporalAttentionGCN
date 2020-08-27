import pickle
import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, choices=['ntu60_xsub', 'ntu60_xview', 'ntu120_xsub', 'ntu120_xsetup'])
parser.add_argument('--score1', default=None)
parser.add_argument('--score2', default=None)
parser.add_argument('--score3', default=None)
args = parser.parse_args()

# load label
if args.dataset == 'ntu60_xsub':
    label_path = open('Data/NTU-RGB+D60/xsub/test_label.pkl', 'rb')
    log_path = 'Log/NTU-RGB+D60/xsub/'
elif args.dataset == 'ntu60_xview':
    label_path = open('Data/NTU-RGB+D60/xview/test_label.pkl', 'rb')
    log_path = 'Log/NTU-RGB+D60/xview/'
elif args.dataset == 'ntu120_xsub':
    label_path = open('Data/NTU-RGB+D120/xsub/test_label.pkl', 'rb')
    log_path = 'Log/NTU-RGB+D120/xsub/'
elif args.dataset == 'ntu120_xsetup':
    label_path = open('Data/NTU-RGB+D120/xsetup/test_label.pkl', 'rb')
    log_path = 'Log/NTU-RGB+D120/xsetup/'
label = np.array(pickle.load(label_path))

# load score
s1_path = (log_path + args.score1 + '/test_score.pkl')
s1 = open(s1_path, 'rb')
s1 = list(pickle.load(s1).items())
s2_path = (log_path + args.score2 + '/test_score.pkl')
s2 = open(s2_path, 'rb')
s2 = list(pickle.load(s2).items())
if args.score3 is not None:
    s3_path = (log_path + args.score3 + '/test_score.pkl')
    s3 = open(s3_path, 'rb')
    s3 = list(pickle.load(s3).items())

# main process
right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, s11 = s1[i]
    _, s22 = s2[i]
    s = S11 + s22
    if args.score3 is not None:
        _, s33 = s3[i]
        s += s3
    rank_5 = s.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    s = np.argmax(s)
    right_num += int(s == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num

print('Top1 accuracy: {:.3f}'.format(100. * acc))
print('Top5 accuracy: {:.3f}'.format(100. * acc5))
print('\n# bye :)\n')