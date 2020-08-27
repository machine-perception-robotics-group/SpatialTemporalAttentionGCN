import torch
import torch.nn as nn

from Tools.Model.Utils.make_graph import Graph
from Tools.Model.Utils.feature_extractor import Feature_extractor
from Tools.Model.Utils.attention_branch import Attention_branch
from Tools.Model.Utils.perception_branch import Perception_branch


class STA_GCN(nn.Module):
    def __init__(self,
                 num_class,
                 in_channels,
                 residual,
                 dropout,
                 num_person,
                 t_kernel_size,
                 layout,
                 strategy,
                 hop_size,
                 num_att_A):
        super().__init__()

        # Graph
        graph = Graph(layout=layout, strategy=strategy, hop_size=hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # config
        kwargs = dict(s_kernel_size=A.size(0),
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A.size())

        # Feature Extractor
        f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 32, 1], [32, 64, 2], [64, 64, 1]]
        self.feature_extractor = Feature_extractor(config=f_config, num_person=num_person, **kwargs)

        # Attention Branch
        a_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.attention_branch = Attention_branch(config=a_config, num_class=num_class, num_att_A=num_att_A, **kwargs)

        # Perception Branch
        p_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.perception_branch = Perception_branch(config=p_config, num_class=num_class, num_att_A=num_att_A, **kwargs)

    def forward(self, x):

        N, c, t, v, M = x.size()

        # Feature Extractor
        feature = self.feature_extractor(x, self.A)

        # Attention Branch
        output_ab, att_node, att_A = self.attention_branch(feature, self.A, N, M)

        # Attention Mechanism
        att_x = feature * att_node

        # Perception Branch
        output_pb = self.perception_branch(att_x, self.A, att_A, N, M)

        return output_ab, output_pb, att_node, att_A
