import argparse
import logging
import os
import os.path as osp
import sys
import random
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from parallel import DataParallelModel, DataParallelCriterion
from torch.nn.parallel.scatter_gather import gather
import dataset, metric, loss
from network.ABiU_Net import VisionTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='ABiU_Net', help='Model name')
parser.add_argument('--data_dir',type=str,default='dataset',help='data directory')
parser.add_argument('--save_dir', default='./result', help='directory to save the results')
parser.add_argument('--max_epochs',type=int,default=50,help='maximum epoch number to train')
parser.add_argument('--batch_size',type=int,default=4,help='batch_size')
parser.add_argument('--iter_size',type=int,default=4,help='iter_size')
parser.add_argument('--num_workers',type=int,default=4,help='number of workers to load data')
parser.add_argument('--base_lr',type=float,default=5e-5,help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,default=384,help='input patch size of network input')
parser.add_argument('--resume', type=str,default='',help='resume network')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
args = parser.parse_args()


def main():
    saveDir = args.save_dir + '_epoch' + str(args.max_epochs) + '/'
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    saveDir = saveDir + str(args.model_name) + \
            '_batch' + str(args.batch_size) + \
            '_iter' + str(args.iter_size) + \
            '_lr' + str(args.base_lr)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    log_file = os.path.join(saveDir, 'Log_'+args.model_name+'.txt')
    if os.path.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
        logger.write('\n%s\t\t%s\t%s\t%s\t%s\t%s\t\t%s\t%s\tlr' % ('Epoch', \
                                                                   'Loss(Tr)', 'SegLoss(Tr)', 'DualLoss(Tr)', \
                                                                   'F_beta(tr)', 'MAE(tr)', 'F_beta(val)', 'MAE(val)'))
    logger.flush()

    model = VisionTransformer()
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    trainloader, valloader = dataset.setup_loaders(args)
    max_batches = len(trainloader)
    criterion = loss.get_loss()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(num_params / 1048576))

    if torch.cuda.device_count() > 1:
        model = DataParallelModel(model)
    model = model.cuda()

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    maxFbeta = 0
    maxEpoch = 0
    for epoch_num in range(start_epoch, args.max_epochs):
        cur_iter = 0
        lossTr, seglossTr, duallossTr, F_beta_tr, MAE_tr, lr = \
            train(trainloader, model, criterion, optimizer, epoch_num, max_batches, cur_iter)
        lossVal, F_beta_val, MAE_val = validate(valloader, model, criterion)

        torch.save({
            'epoch': epoch_num + 1,
            'arch': str(args.model_name),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'seglossTr': seglossTr,
            'duallossTr': duallossTr,
            'F_beta_tr': F_beta_tr,
            'F_beta_val': F_beta_val,
            'lr': lr
        }, os.path.join(saveDir, args.model_name + '.pth.tar'))

        model_file_name = os.path.join(saveDir, args.model_name + '_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), model_file_name)

        logger.write('\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f' \
            % (epoch_num, lossTr, seglossTr, duallossTr, F_beta_tr, MAE_tr, F_beta_val, MAE_val, lr))
        logger.flush()
        print('\nEpoch ' + str(epoch_num) + ': Details')
        print('\nEpoch No. %d:\tTrain Loss = %.4f\tSegLoss = %.4f\tDualLoss = %.4f\t'
            'Val Loss = %.4f\t F_beta(tr) = %.4f\t F_beta(val) = %.4f' \
            % (epoch_num, lossTr, seglossTr, duallossTr, lossVal, F_beta_tr, F_beta_val))
        if F_beta_val >= maxFbeta:
            maxFbeta = F_beta_val
            maxEpoch = epoch_num
        torch.cuda.empty_cache()
    logger.flush()
    logger.close()

    log_best = os.path.join(saveDir, 'Best_'+args.model_name+'.txt')
    with open(log_best, 'a+') as log_best:
        log_best.write("\nmaxEpoch: %d\t maxF_beta: %.4f" % (maxEpoch, maxFbeta))


def train(trainloader, model, criterion, optimizer, epoch_num, max_batches, cur_iter):
    model.train()
    salEvalTrain = metric.SalEval()
    counter = 0

    train_main_loss = AverageMeter()
    train_seg_loss = AverageMeter()
    train_dual_loss = AverageMeter()

    for i_batch, sampled_batch in enumerate(trainloader):
        lr = adjust_learning_rate(args, optimizer, epoch_num, i_batch+cur_iter, max_batches)

        inputs, mask = sampled_batch
        inputs, mask = inputs.cuda(), mask.cuda()

        outputs = model(inputs)
        '''
        if torch.cuda.device_count() <= 1:
            pred1, pred2, pred3, pred4, pred5, pred6, pred7 = tuple(outputs)
            loss = criterion(pred1, pred2, pred3, pred4, pred5, pred6, pred7, mask)
        else:
            loss = criterion(outputs, mask)
        '''
        main_loss = None
        loss_dict = None

        loss_dict = criterion(outputs, mask)

        log_seg_loss = loss_dict['seg_loss'].mean().clone().detach_()
        train_seg_loss.update(log_seg_loss.item())
        if main_loss is not None:
            main_loss += loss_dict['seg_loss']
        else:
            main_loss = loss_dict['seg_loss']
        log_dual_loss = loss_dict['dual_loss'].mean().clone().detach_()
        train_dual_loss.update(log_dual_loss.item())
        if main_loss is not None:
            main_loss += loss_dict['dual_loss']
        else:
            main_loss = loss_dict['dual_loss']

        main_loss = main_loss.mean()
        log_main_loss = main_loss.clone().detach_()

        train_main_loss.update(log_main_loss.data.item())

        counter = counter + 1
        main_loss = main_loss / args.iter_size
        main_loss.backward(retain_graph=True)

        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        if torch.cuda.device_count() > 1:
            outputs = gather(outputs, 0, dim=0)[0]
        else:
            outputs = outputs[0]
        salEvalTrain.add_batch(outputs.squeeze(1).data.cpu().numpy(), mask.data.cpu().numpy())

        if i_batch % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch_num, args.max_epochs, i_batch, len(trainloader)) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=train_main_loss) + \
                   'Seg Loss {segloss.val:f} (avg:{segloss.avg:f}) '.format(segloss=train_seg_loss) + \
                   'Dual Loss {dualloss.val:f} (avg:{dualloss.avg:f}) '.format(dualloss=train_dual_loss)
            print(info)
    F_beta, MAE = salEvalTrain.get_metric()
    lr = optimizer.param_groups[-1]['lr']
    return train_main_loss.avg, train_seg_loss.avg, train_dual_loss.avg, F_beta, MAE, lr


def validate(valloader, model, criterion):
    model.eval()
    salEvalVal = metric.SalEval()
    val_loss = AverageMeter()

    for vi, data in enumerate(valloader):
        inputs, mask = data
        inputs, mask = inputs.cuda(), mask.cuda()

        with torch.no_grad():
            outputs = model(inputs)

        torch.cuda.synchronize()

        loss = criterion(outputs, mask)
        val_loss.update(sum(loss.values()).item())

        if torch.cuda.device_count() > 1:
            outputs = gather(outputs, 0, dim=0)[0]
        else:
            outputs = outputs[0]

        salMap_numpy = outputs.squeeze(1).data.cpu().numpy()
        salEvalVal.add_batch(salMap_numpy, mask.data.cpu().numpy())
        #salMap_save = (salMap_numpy[0]*255).astype(np.uint8)
        #cv2.imwrite(osp.join('test', str(vi) + '.png'), salMap_save)

        info = '[{0}]'.format(vi) + \
                'Val Loss {valloss.val:f} (avg:{valloss.avg:f})'.format(valloss=val_loss)
        print(info)

    F_beta, MAE = salEvalVal.get_metric()
    return val_loss.avg, F_beta, MAE


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    cur_iter = max_batches*epoch + iter
    max_iter = max_batches*args.max_epochs
    lr = args.base_lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + val
        self.count = self.count + 1
        self.avg = self.sum / self.count


if __name__ == "__main__":
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    main()
