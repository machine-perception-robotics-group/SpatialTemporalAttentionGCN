import os
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
from Tools.Model.sta_gcn import STA_GCN as STA_GCN
from Tools.Feeder.feeder import Feeder as Feeder
import Tools.Visualize.visualize_tools as visualize_tools


parser = argparse.ArgumentParser(description='Main config')
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--visualize', action='store_true')
par = parser.parse_args()

# load config
with open(par.config, 'r') as f:
    args = yaml.load(f, Loader=yaml.SafeLoader)

log_dir = args.pop('log_dir')
device = args.pop('device')
output_device = device[0] if type(device) is list else device
torch.backends.cudnn.enabled = False

# model
model_args = args.pop('model_args')
num_class = model_args['num_class']
num_att_A = model_args['num_att_A']
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

# load data
feeder_args = args.pop('feeder_args')
test_feeder_args = feeder_args.pop('test_feeder_args')
frame_size = feeder_args.pop('frame_size')
data_loader = torch.utils.data.DataLoader(dataset=Feeder(data_path=test_feeder_args['data_path'],
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
                                          batch_size=feeder_args['batch_size'] * 2,
                                          shuffle=False,
                                          num_workers=0)


correct_ab = 0
correct_pb = 0
score_list = []
confusion_matrix = np.zeros((num_class, num_class))
vis_dir = log_dir + '/visualize'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

with torch.no_grad():
    for batch_idx, (data, label, name) in enumerate(tqdm(data_loader, leave=False, desc='#  test')):
        data = data.to(output_device)
        label = label.to(output_device)

        output_ab, output_pb, att_node, att_A = model(data)

        _, predict = torch.max(output_ab.data, 1)
        correct_ab += (predict == label).sum().item()
        _, predict = torch.max(output_pb.data, 1)
        correct_pb += (predict == label).sum().item()

        # confusion matrix
        for l, p in zip(label.view(-1), predict.view(-1)):
            confusion_matrix[l.long(), p.long()] += 1

        # score
        score_list.append(output_pb.data.cpu().numpy())

        # visualize
        if par.visualize:
            visualize_tools.visualizing(att_node, att_A, data.cpu().numpy(), label.cpu().numpy(),
                                        predict.cpu().numpy(),
                                        name, frame_size, num_att_A, vis_dir, args['dataset'])

# save score
print('# save score: {}/test_score.pkl'.format(log_dir))
score = np.concatenate(score_list)
score_dict = dict(zip(data_loader.dataset.sample_name, score))
with open('{}/test_score.pkl'.format(log_dir), 'wb') as f:
    pickle.dump(score_dict, f)

# confusion matrix
len_cm = len(confusion_matrix)
for i in range(len_cm):
    sum_cm = np.sum(confusion_matrix[i])
    for j in range(len_cm):
        confusion_matrix[i][j] = 100 * (confusion_matrix[i][j] / sum_cm)
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
figname = os.path.join(vis_dir, 'confusion_matrix.pdf')
plt.savefig(figname, bbox_inches='tight')
plt.close()

len_data_loader = len(data_loader.dataset)
print('# accuracy attention  branch: {:.3f}'.format((100. * correct_ab /len_data_loader)))
print('# accuracy perception branch: {:.3f}'.format((100. * correct_pb /len_data_loader)))
