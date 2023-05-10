import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
from os.path import join
import numpy as np

def load_data(args):
    # 读取数据
    datafold = os.path.abspath(join('data')) 
    train_set, test_set, feature_num, item_feature, user_feature = pickle.load(open(join(datafold, 'all_data.pkl'),'rb'))
    
    # 保存特征数量以及用户特征维度    
    args.feature_num = feature_num
    args.user_feat_dim = len(user_feature[0])
    
    # 构建Dataset&DataLoader
    train_dataset = MyDataset(train_set, item_feature, user_feature)
    test_dataset = MyDataset(test_set, item_feature, user_feature)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=1,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)
    
    return train_loader, test_loader
    

class MyDataset(Dataset):
    def __init__(self, data, item_feature, user_feature)  -> None:
        
        super().__init__()
        self.data = data
        self.item_feature = item_feature
        self.user_feature = user_feature
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        userid, itemid, label = self.data[index]
        feature = self.item_feature[itemid] # 包含了所有item_id
        feature['user_id'] = userid
        feature['user_feature'] = self.user_feature[userid]
        
        return feature, label
        
def collate_fn(batch_data):
    all_fatures =  {'cat_id': [],
                    'seller_id': [],
                    'item_id': [],
                    'brand_id': [],
                    'user_id': [],
                    'user_feature': []}
    all_labels = []
    
    for feature, label in batch_data:
        for k, v in feature.items():
            all_fatures[k].append(v)
        all_labels.append(label)
    
    all_fatures['cat_id'] = torch.LongTensor(all_fatures['cat_id'])
    all_fatures['seller_id'] = torch.LongTensor(all_fatures['seller_id'])
    all_fatures['item_id'] = torch.LongTensor(all_fatures['item_id'])
    all_fatures['brand_id'] = torch.LongTensor(all_fatures['brand_id'])
    all_fatures['user_id'] = torch.LongTensor(all_fatures['user_id'])
    
    all_fatures['user_feature'] = torch.FloatTensor(np.stack(all_fatures['user_feature'],0))
    
    all_labels = torch.FloatTensor(all_labels)
    
    return {
            "feature": all_fatures,
            "label": all_labels
            }