import numpy as np
import torch
import os
import torch.nn as nn
from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from dataset import grading_dataset
from datetime import datetime
from tqdm import tqdm
from functions import progress_bar
from torchnet import meter
from sklearn.metrics import f1_score,roc_auc_score,cohen_kappa_score,accuracy_score,confusion_matrix
from lr_scheduler import LRScheduler
import random
import torch.backends.cudnn as cudnn
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

best_kappa_global = 0

class grading_model():
    def __init__(self, args):
        self.args = args


    def run(self):
        # global best_kappa
        global save_dir

        # if self.args.model == 'effb6':
        #     net = timm.create_model('tf_efficientnet_b6', pretrained=True, num_classes=self.args.n_classes)
        # elif self.args.model == 'incep_res':
        #     net = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=self.args.n_classes)
        # elif self.args.model == 'mobile':
        #     net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=self.args.n_classes)
        # elif self.args.model == 'incepv3':
        #     net = timm.create_model('tf_inception_v3', pretrained=True, num_classes=self.args.n_classes)
        # elif self.args.model == 'vit':
        #     net = timm.create_model('vit_small_r26_s32_384', pretrained=True, num_classes=self.args.n_classes)
        # # elif self.args.model == 'resnest50':
        # else:
        #     net = timm.create_model('resnest50d', pretrained=True, num_classes=self.args.n_classes)
        net = timm.create_model('vit_small_r26_s32_384', pretrained=True, num_classes=self.args.n_classes)

        if self.args.pretrained:
            if self.args.model == 'vit':
                ckpt = torch.load('checkpoints/ddr_resnest50_n3/35.pkl')
            state_dict = ckpt['net']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            net.load_state_dict(unParalled_state_dict, True)

        net = nn.DataParallel(net)
        net = net.cuda()

        trainset = grading_dataset(train=True, val=False, test=False)#, KK=self.args.KK
        valset = grading_dataset(train=False, val=True, test=False)#, KK=self.args.KK

        drop_last = False
        if (self.args.model == 'resnest50' and self.args.KK == 4) or (self.args.model == 'vit' and (self.args.KK == 2 or self.args.KK == 3)):
            drop_last = True

        train_loader = DataLoader(trainset, shuffle=True, batch_size=self.args.batch_size, num_workers=8, pin_memory=True, drop_last=drop_last)

        val_loader = DataLoader(valset, shuffle=False, batch_size=self.args.batch_size, num_workers=4, pin_memory=True)

        # optim & crit
        if self.args.model == 'seresnext101':
            optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        else:
            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)  # 1e-5
        lr_scheduler = LRScheduler(optimizer, len(train_loader), self.args)

        if self.args.pretrained:
            weight = None
        else:
            weight = torch.tensor([0.1379, 0.2140, 0.6481])
        criterion = nn.CrossEntropyLoss(weight=weight)
        criterion = criterion.cuda()
        con_matx = meter.ConfusionMeter(self.args.n_classes)

        criterion_mix = SoftTargetCrossEntropy().cuda()
        mixup_fn = Mixup(mixup_alpha=0.4, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=0.5, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=3)

        save_dir = './checkpoints/'+ self.args.visname + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test_log_dir = '/logs/' 
        if not os.path.exists(test_log_dir):
            os.makedirs(test_log_dir)
        test_log = open(test_log_dir + self.args.visname + '.txt', 'w')

        start_epoch = 0

        for epoch in range(start_epoch, self.args.epochs):
            con_matx.reset()
            net.train()
            total_loss = .0
            total = .0
            correct = .0
            count = .0

            for i, (x, label, _) in enumerate(train_loader):
                lr = lr_scheduler.update(i, epoch)
                x = x.float().cuda()
                label = label.cuda()

                y_pred = net(x)

                loss_clf = criterion(y_pred, label)
                prediction = y_pred.max(1)[1]
                loss = loss_clf

                total_loss += loss.item()
                total += x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct += prediction.eq(label).sum().item()
                count += 1

                progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f ' % (total_loss / (i + 1), 100. * correct / total))

            test_log.write('Epoch:%d  lr:%.5f  Loss:%.4f  Acc:%.4f \n' % (epoch, lr, total_loss / count, correct / total))
            test_log.flush()
            self.validate_model(net, val_loader, epoch, test_log, optimizer)

    # @torch.no_grad()
    def validate_model(self, net, val_loader, epoch, test_log, save_dir):
        global best_kappa_global

        net.eval()
        total_samples = 0
        confusion_matrix = meter.ConfusionMeter(self.args.n_classes)

        pred_labels = []
        predicted_labels = []
        true_labels = []

        for i, (inputs, labels, _) in enumerate(val_loader):
            inputs = inputs.float().cuda()
            labels = labels.cuda()

            outputs = net(inputs)
            confusion_matrix.add(outputs.detach(), labels.detach())

            predicted_labels.extend(outputs.squeeze(-1).cpu().detach())
            true_labels.extend(labels.cpu().detach())

            predictions = outputs.max(1)[1]
            pred_labels.extend(predictions.cpu().detach())

            total_samples += inputs.size(0)

            progress_bar(i, len(val_loader))

        kappa = cohen_kappa_score(np.array(true_labels), np.array(pred_labels), weights='quadratic')
        accuracy = accuracy_score(np.array(true_labels), np.array(pred_labels))

        print('Validation Epoch:', epoch, ' Accuracy:', accuracy, 'Kappa:', kappa, 'Confusion Matrix:',
              str(confusion_matrix.value()))
        test_log.write('Validation Epoch:%d   Accuracy:%.4f   Kappa:%.4f  Confusion Matrix:%s \n' % (
        epoch, accuracy, kappa, str(confusion_matrix.value())))
        test_log.flush()

        if kappa > best_kappa_global:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'kappa': kappa,
                'epoch': epoch
                # 'optimizer': optimizer
            }
            save_path = os.path.join(save_dir, str(epoch) + '.pkl')
            torch.save(state, save_path)
            best_kappa_global = kappa