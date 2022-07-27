import sys
sys.path.insert(0, '.')

import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from metric import SalEval
from network.ABiU_Net import VisionTransformer


def get_mean_set(args):
    # for DUTS training dataset
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    return mean, std

@torch.no_grad()
def validateModel(args, model, dataset, image_list, label_list):
    mean, std = get_mean_set(args)
    SalEvalVal = SalEval()
    for idx in range(len(image_list)):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255
        label = label.astype(dtype=np.bool)
        img = image.astype(np.float32)
        img = img / 255
        img = img[:, :, ::-1]
        img -= mean
        img /= std
        img = cv2.resize(img, (args.img_size, args.img_size))
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        img_variable = Variable(img_tensor)
        if args.gpu:
            img_variable = img_variable.cuda()

        start_time = time.time()
        img_out = model(img_variable)[0]
        torch.cuda.synchronize()
        diff_time = time.time() - start_time
        #print('Segmentation for {}/{} takes {:.3f}s per image'.format(idx, len(image_list), diff_time))

        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        if (img_out.max()-img_out.min()) == 0:
            continue
        img_out = (img_out-img_out.min()) / (img_out.max()-img_out.min())
        salMap_numpy = img_out.squeeze(1).data.cpu().numpy()
        SalEvalVal.add_batch(salMap_numpy, np.expand_dims(label, 0))

        salMap_numpy = (salMap_numpy[0]*255).astype(np.uint8)
        name = image_list[idx].split('/')[-1]
        if not osp.isdir(osp.join(args.savedir, dataset)):
            os.mkdir(osp.join(args.savedir, dataset))
        if not osp.isdir(osp.join(args.savedir, dataset, args.model_name)):
            os.mkdir(osp.join(args.savedir, dataset, args.model_name))
        cv2.imwrite(osp.join(args.savedir, dataset, args.model_name, name[:-4] + '.png'), salMap_numpy)

    F_beta, MAE = SalEvalVal.get_metric()
    print(dataset, args.model_name, '\nOverall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))


def main(args):
    model = VisionTransformer()
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(args.pretrained)
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        new_keys.append(key.replace('module.', ''))
        new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict, strict=True)

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    if not osp.isdir(args.savedir):
        os.mkdir(args.savedir)

    # read all the images in the folder
    for dataset in ['SOD',  'HKU-IS', 'ECSSD', 'DUT-OMRON','THUR15K', 'DUTS-TE']:#
        image_list = list()
        label_list = list()
        with open(osp.join(args.data_dir, dataset+'.lst')) as textFile:
            for line in textFile:
                line_arr = line.split()
                image_list.append(args.data_dir + line_arr[0].strip())
                label_list.append(args.data_dir + line_arr[1].strip())

        validateModel(args, model, dataset, image_list, label_list)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='ABiU_Net', help='Model name')
    parser.add_argument('--data_dir',type=str,default='dataset',help='data directory')
    #parser.add_argument('--file_list', default='SOD.lst', help='Data directory')
    parser.add_argument('--img_size', type=int,default=384,help='input patch size of network input')
    parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default='./result_epoch50/ABiU_Net_50.pth', help='Pretrained model')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
