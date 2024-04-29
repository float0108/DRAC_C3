import numpy as np
import torch
import argparse
import torch.nn as nn
from datetime import datetime
import random
import torch.backends.cudnn as cudnn
from model import grading_model

#TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 
    

def parse_args():
    global args
    args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='res50', help='model')
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=3, type=int, help='n-classes')
parser.add_argument('--pretrained', '-pre', default=False, type=bool, help='use pretrained model')
parser.add_argument('--dataset', '-data', default='pair', type=str, help='dataset')
parser.add_argument('--KK', '-KK', default=0, type=int, help='KFold')
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument("--lambda_value", default=0.25, type=float)


best_acc = 0
best_kappa = 0


if __name__ == '__main__':
    args = parser.parse_args()
    model = grading_model(args)
    model.run()
    print(3)
