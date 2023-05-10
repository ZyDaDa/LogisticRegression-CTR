from tqdm import tqdm
from dataset import load_data
import torch
from parse import get_parse
from utils import fix_seed, metrics
from model import Model
import numpy as np

SHOW = True

def main(args):
    fix_seed()

    train_loader, test_loader = load_data(args)
    
    # 定义模型
    model = Model(args)
    model.to(args.device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)

    for e in range(args.epoch):
        # 训练模型
        model.train()
        all_loss = 0.0
        all_num = 0
        if SHOW:
            bar = tqdm(train_loader, total=len(train_loader),ncols=100)
        else:
            bar = train_loader
        for data in bar:
            pred = model(dict((k,d.to(args.device)) for k,d in data['feature'].items()))
            loss = model.loss_func(pred, data['label'].to(args.device))
            
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            all_num += len(data['label'])
            if SHOW:
                bar.set_postfix(Epoch=e, LR=optimizer.param_groups[0]['lr'], Train_Loss=all_loss/all_num)

        # 测试模型
        y_true = []
        y_pre = []
        for data in tqdm(test_loader,ncols=80,desc='test'):
            scores = model(dict((k,d.to(args.device)) for k,d in data['feature'].items()))

            y_true.append(data['label'].numpy())
            y_pre.append(scores.detach().cpu().numpy())
        # 拼接所有测试数据
        y_true = np.concatenate(y_true)
        y_pre = np.concatenate(y_pre)
        results = metrics(y_true, y_pre)
        # 计算并输出评价指标
        for k, v in results.items():
            print("%s\t%.4f"%(k,v))

if __name__ == '__main__':
    args = get_parse() # 获取超参数e
    main(args)

   