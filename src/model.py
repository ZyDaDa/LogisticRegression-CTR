import torch
import torch.nn as nn
import math


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 定义特征的Embedding矩阵
        emb_dict = {}
        all_dim = args.user_feat_dim
        for k, n in args.feature_num.items():
            dim = int(math.log(n)+8) # 计算每个特征维度
            all_dim += dim  # 总维度
            emb_dict[k] = nn.Embedding(n, dim) # 当前特征的Embedding
        
        self.embeddings = nn.ModuleDict(emb_dict) # 全部特征保存为一个dict
        
        # 定义逻辑回归模型
        self.LogisticsRegression = nn.Sequential(
            nn.Linear(all_dim, 1, bias=True),
            nn.Sigmoid()
        )
        
        # 参数初始化
        for p in self.parameters():
            nn.init.normal_(p.data, 0, 0.1)

        self.loss_func = nn.BCELoss()

    def forward(self, features):
        
        # 特征拼接
        all_feature = [features['user_feature']]
        for k, emb in self.embeddings.items():
            all_feature.append(emb(features[k]))
        all_feature = torch.concat(all_feature, -1)
        
        # 逻辑回归
        pred = self.LogisticsRegression(all_feature).squeeze()
        
        return pred


                
        
