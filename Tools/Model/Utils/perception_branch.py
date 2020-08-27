import torch
import torch.nn as nn
import torch.nn.functional as F

from Tools.Model.Utils.graph_convolution import Stgc_block


class Perception_branch(nn.Module):
    def __init__(self,
                 config,
                 num_class,
                 num_att_A,
                 s_kernel_size,
                 t_kernel_size,
                 dropout,
                 residual,
                 A_size,
                 use_att_A=True):
        super().__init__()

        # STGC Block
        kwargs = dict(s_kernel_size=s_kernel_size,
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A_size,
                      use_att_A=use_att_A,
                      num_att_A=num_att_A)
        self.stgc_block0 = Stgc_block(config[0][0], config[0][1], config[0][2], **kwargs)
        self.stgc_block1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
        self.stgc_block2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
        self.stgc_block3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
        self.stgc_block4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

        # prediction
        self.fc = nn.Conv2d(config[-1][1], num_class, kernel_size=1, padding=0)

    def forward(self, x, A, att_A, N, M):

        # STGC Block
        x = self.stgc_block0(x, A, att_A)
        x = self.stgc_block1(x, A, att_A)
        x = self.stgc_block2(x, A, att_A)
        x = self.stgc_block3(x, A, att_A)
        x = self.stgc_block4(x, A, att_A)

        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
