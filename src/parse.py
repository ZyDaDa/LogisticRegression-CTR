import argparse
import torch

def get_parse():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--device', default='cuda', type=str,help='cuda or cpu')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') 
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=8, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=0, help='l2 penalty') 
    
    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: args.device = torch.device('cpu')
    return args
