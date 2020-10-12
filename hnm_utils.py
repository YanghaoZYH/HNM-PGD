import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from mmdet.core import tensor2imgs,bbox2roi, multiclass_nms
import numpy as np
import os
from tool.darknet2pytorch import *
from tqdm import tqdm
from skimage import measure

from utils.utils import *
import matplotlib.pyplot as plt

device='cuda:0'

original_image_path = './select1000_new'
output_image_path = './select1000_new_p'
yoloV4_cfgfile = "models/yolov4.cfg"
yoloV4_weightfile = "models/yolov4.weights"

tau = 8
pgd_step = 128
obj_conf_thresh = 0.4
alpha = 1/255.

anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
num_anchors = 9
num_classes = 80
anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
strides = [8, 16, 32]
anchor_step = len(anchors) // num_anchors

def getmm_list(detector,image,img_metas):
    x_feat = detector.extract_feat(image)

    proposal_list = detector.rpn_head.simple_test_rpn(x_feat, img_metas)

    img_shape = img_metas[0]['img_shape']
    # img_shape = img_metas[0]['ori_shape']
    rois = bbox2roi(proposal_list)
    bbox_results = detector.roi_head._bbox_forward(x_feat, rois)
    bbox_pred = bbox_results['bbox_pred']
    cls_score = bbox_results['cls_score']
    if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

    scores = F.softmax(cls_score, dim=1) if cls_score is not None else None 
    zeros = torch.zeros_like(scores)
    cls_score = torch.where(scores>0.3,scores,zeros)
    scores = scores[:, :-1]
    valid_mask = scores > 0.1
    scores = torch.masked_select(scores, valid_mask)
    return scores,cls_score

def Image2tensor(img):
    width = img.width
    height = img.height
    tmp_img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    tmp_img = tmp_img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    tmp_img = tmp_img.view(1, 3, height, width)
    tmp_img = tmp_img.float().div(255.0)
    tmp_img = tmp_img.to(device)
    return tmp_img

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def count_patch(metrix, thresh):
    metrix =metrix.cpu().detach()
    ones = torch.FloatTensor(metrix.size()).fill_(1).cpu()
    zeros = torch.FloatTensor(metrix.size()).fill_(0).cpu()
    input_map_new = torch.where((metrix > thresh.cpu()), ones, zeros)
    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    pixel_number = torch.sum(input_map_new)
    print(f'threshold: {thresh}, patch number: {label_max_number}, pixel number: {pixel_number}')
    return input_map_new,label_max_number,pixel_number


def knn_filter(x, size=5, k=13):
    kernel = torch.ones((size, size)).unsqueeze_(0)
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    x = torch.where((x != 0), ones, zeros)
    x.unsqueeze_(0).unsqueeze_(0)
    assert len(x.shape) == 4
    x = F.conv2d(x, kernel.unsqueeze_(0), bias=None, stride=1, padding=size//2)

    x = torch.where((x.squeeze_() > k), ones, zeros)
    return x

def knn_process_small(x, k1 = 13, k2 = 3, pixel_thresh=1000, patch_thresh=10):
    knn_mask = x.cpu().detach()
    success = 0
    while k1 > k2:
        knn_mask = knn_filter(knn_mask,k1,int(k1*k1/2))
        knn_mask = knn_filter(knn_mask,k2,int(k2*k2/2))

        labels = measure.label(knn_mask.numpy()[:, :], background=0, connectivity=2)
        num_patch = np.max(labels)
        num_pixel = torch.sum(knn_mask)
        print(f"patch: {num_patch}, pixel: {num_pixel}")
        if num_pixel <= pixel_thresh and num_patch <= patch_thresh:
            print('success')
            success = 1
            break
        k1-=4
    # plt.imshow(knn_mask.cpu().detach().numpy())
    # plt.savefig('./knn_mask.jpg')
    return knn_mask,success


def get_mask_all_small(num_patch,img_500, img_mmd_800, model,detector, img_metas, num_std, step=100):
    ## using bce loss and l1 loss
    sigm = nn.Sigmoid()
    bceloss = nn.BCELoss(reduction='mean')
    l1_loss = nn.L1Loss()
    lossce = nn.NLLLoss(reduction='mean')
    ## define upsample
    up_sample = torch.nn.Upsample(size=608, mode='bilinear')
    uprcnn = torch.nn.Upsample(size=800, mode='bilinear')

    dataset_mean = img_metas[0]['img_norm_cfg']['mean']/255
    dataset_std = img_metas[0]['img_norm_cfg']['std']/255

    mu = torch.Tensor((dataset_mean)).unsqueeze(-1).unsqueeze(-1).cuda()
    std = torch.Tensor((dataset_std)).unsqueeze(-1).unsqueeze(-1).cuda()
    unnormalize = lambda x: x*std + mu
    normalize = lambda x: (x-mu)/std

    ## init mix gradient matrix
    mix_grad = torch.zeros_like(img_500).to(device)
    
    for p in range(step):
        delta = torch.zeros_like(img_500).to(device).normal_(0, tau/255.).to(device)

        delta.data = clamp(delta, 0 - img_500, 1 - img_500)
        delta.requires_grad = True

        image = normalize(unnormalize(img_mmd_800)+uprcnn(delta))

        obj_loss = 0
        obj_confs_rcnn,bk_scores = getmm_list(detector,image,img_metas)

        obj_confs_rcnn = obj_confs_rcnn[obj_confs_rcnn > 0.1 ]

        ## make detection
        list_boxes = model(up_sample(img_500 + delta))

        ## get all object confidence
        obj_confs_list = []
        for idx, boxx in enumerate(list_boxes):
            obj_confs_list.append(torch.cat((boxx[0][4].view(-1),boxx[0][4+85].view(-1),boxx[0][4+170].view(-1)),0))
        obj_confs = torch.cat([obj_confs_list[i] for i in range(len(obj_confs_list))],0)

        ## compute loss regarding where confidence is larger than threshold
        obj_confs = sigm(obj_confs)
        obj_confs = obj_confs[obj_confs > 0.3]
        targets = torch.ones_like(obj_confs).to(device)
        obj_loss += (bceloss(obj_confs, targets) )*(len(obj_confs)/(len(obj_confs_rcnn)+len(obj_confs)))

        if len(obj_confs_rcnn):
            targets_rcnn = torch.ones_like(obj_confs_rcnn).to(device)
            obj_loss += (bceloss(obj_confs_rcnn,targets_rcnn))*(len(obj_confs_rcnn)/(len(obj_confs_rcnn)+len(obj_confs)))

        fg_scores = bk_scores[bk_scores.max(1)[1]!=80]
        targets_fg = torch.LongTensor([80 for i in range(len(fg_scores))]).to(device)
        obj_loss -=lossce(fg_scores,targets_fg)

        obj_loss.backward()
        grad = delta.grad.detach()
        mix_grad += grad
        delta.grad.zero_()


    sensitivity_matrix = ((torch.sum(torch.abs(mix_grad), axis=1)).squeeze_())
    # print(sensitivity_matrix.shape)
    heat_map,_,_ = count_patch(sensitivity_matrix, thresh=torch.mean(sensitivity_matrix)+num_std*torch.std(sensitivity_matrix))
    pgd_mask, flag = knn_process_small(heat_map,pixel_thresh=num_patch)
    return pgd_mask,flag

