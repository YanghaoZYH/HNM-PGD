from PIL import ImageFile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
from tool.darknet2pytorch import *
from skimage import measure
from utils.utils import *
sys.path.append('../mmdetection/')
from mmdet import __version__
from mmdet.apis import init_detector,inference_detector
from mmdet.apis.inference import LoadImage
import warnings
from mmdet.core import tensor2imgs
import argparse
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import tensor2imgs,bbox2roi, multiclass_nms
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv import Config, DictAction
from mmdet.datasets import build_dataloader, build_dataset
from hnm_utils import get_mask_all_small, clamp, count_patch, Image2tensor

def init_patch(metrix, thresh):
    metrix =metrix.detach().cpu()
    ones = torch.FloatTensor(metrix.size()).fill_(1).cpu()
    zeros = torch.FloatTensor(metrix.size()).fill_(0).cpu()
    input_map_new = torch.where((metrix > thresh), ones, zeros)
    return input_map_new

def read_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    mask = mask.transpose(2,0,1)
    mask = torch.from_numpy(mask).sum(0)
    mask = init_patch(mask, 0.1)
    return mask


def inference_detector2(model, img_path):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    test_pipeline = [LoadImage()]+ cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=img_path)
    data = test_pipeline(data)
    
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    imgs = data['img'][0]
    img_metas = data['img_metas'][0]
    return imgs, img_metas

def parse_mmd(result_p):
    if isinstance(result_p, tuple):
        bbox_results, _ = result_p
        result_p = bbox_results

    result_p = np.concatenate(result_p)

    result_above_confidence_num_p = 0

    for ir in range(len(result_p)):
        if result_p[ir, 4] > show_score_thr:
            result_above_confidence_num_p = result_above_confidence_num_p + 1
    # print(result_p[:, 4][result_p[:, 4]>0.3])
    return result_above_confidence_num_p


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def getmm_list2(detector,image,img_metas):
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

    scoresall = F.softmax(cls_score, dim=1) if cls_score is not None else None 

    scores = scoresall[:, :-1]

    valid_mask = scores > 0.1

    obj_confs_rcnn = torch.masked_select(scores, scores > 0.05)

    return obj_confs_rcnn, scoresall[torch.sum(valid_mask,1)!=0]

def conntet_test(input_img):

    ones = torch.cuda.FloatTensor(input_img[0].size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img[0].size()).fill_(0)

    input_img_tmp2 = torch.where((input_img[0] != 0), ones, zeros) + \
                     torch.where((input_img[1] != 0), ones, zeros) + \
                     torch.where((input_img[2] != 0), ones, zeros)
    input_map_new = torch.where(input_img_tmp2 > 0, ones, zeros)

    whole_size = input_map_new.shape[0] * input_map_new.shape[1]
    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)

    total_area = torch.sum(input_map_new).item()
    total_area_rate = total_area / whole_size
    if label_max_number>10 or total_area_rate > 0.02:
        return True, label_max_number,total_area_rate,total_area
    else :
        return False,label_max_number,total_area_rate,total_area




config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
mmdmodel = init_detector(config, checkpoint, device='cuda:0')
show_score_thr=0.3

cfg = Config.fromfile(config)

model = cfg.model
train_cfg = cfg.train_cfg
test_cfg = cfg.test_cfg
model['pretrained'] = None
detector = build_detector(model, train_cfg=train_cfg, test_cfg=test_cfg)
device='cuda:0'
if checkpoint is not None:
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(detector, checkpoint, map_location=map_loc)
    if 'CLASSES' in checkpoint['meta']:
        detector.CLASSES = checkpoint['meta']['CLASSES']
    else:
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use COCO classes by default.')
        detector.CLASSES = get_classes('coco')

detector.cfg = cfg
detector.to(device)
detector.eval() 


resize_small = transforms.Compose([
    transforms.Resize((608, 608)),])

center_crop = transforms.Compose([
    transforms.CenterCrop(608)])

resize_back = transforms.Compose([
    transforms.Resize((500, 500)),transforms.ToTensor()])

resize2 = transforms.Compose([
        transforms.ToTensor()])

original_image_path = './select1000_new'
output_image_path = './select1000_new_p'
yoloV4_cfgfile = "models/yolov4.cfg"
yoloV4_weightfile = "models/yolov4.weights"

darknet_model = Darknet(yoloV4_cfgfile)
darknet_model.load_weights(yoloV4_weightfile)
darknet_model = darknet_model.eval().cuda()


anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
num_anchors = 9
num_classes = 80
anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
strides = [8, 16, 32]
anchor_step = len(anchors) // num_anchors



files = os.listdir('select1000_new')
files.sort()
fail_image = []

img_path0 = 'select1000_new/104.png'
_,img_metas = inference_detector2(detector,img_path0)

dataset_mean = img_metas[0]['img_norm_cfg']['mean']/255
dataset_std = img_metas[0]['img_norm_cfg']['std']/255

mu = torch.Tensor((dataset_mean)).unsqueeze(-1).unsqueeze(-1).cuda()
std = torch.Tensor((dataset_std)).unsqueeze(-1).unsqueeze(-1).cuda()
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std

sigm = nn.Sigmoid()
# loss = nn.SmoothL1Loss(reduction='sum')
loss = nn.BCELoss(reduction = 'sum')
lossce = nn.NLLLoss(reduction='sum')
count = 0

alpha= 4/255

uprcnn = torch.nn.Upsample(size=800, mode='bilinear')

upyolo = torch.nn.Upsample(size=608, mode='bilinear')

frcnn_hflip = torch.eye(800).flip(0).to(device)
yolo_hflip = torch.eye(608).flip(0).to(device)


def pad_flip(img_608, img_800,ptb):
    frcnn_img_0 = normalize(unnormalize(img_800)+uprcnn(ptb))
    yolo_img_0 = img_608 + upyolo(ptb)

    # yolo_re_size = np.random.choice(a=range(400,600,2),
    #                                 size=1, replace=False, p=None).item()
    yolo_re_size = 500
    yolo_pad_size = (608 - yolo_re_size) //2
    yolo_pad = torch.nn.ConstantPad2d(padding=yolo_pad_size, value=0.)
    yolo_img_1 = F.pad(F.interpolate(yolo_img_0, size=yolo_re_size, mode="bilinear", align_corners=False),
                        pad=tuple([yolo_pad_size]*4), mode='constant', value=0)

    # frcnn_re_size = np.random.choice(a=range(600,800,2),
    #                                 size=1, replace=False, p=None).item()

    frcnn_re_size = 500
    frcnn_pad_size = (800 - frcnn_re_size) // 2
    frcnn_pad = torch.nn.ConstantPad2d(padding=frcnn_pad_size, value=0.)
    frcnn_img_1 = F.pad(F.interpolate(frcnn_img_0, size=frcnn_re_size, mode="bilinear", align_corners=False),
                        pad=tuple([frcnn_pad_size]*4), mode='constant', value=0)

    # return yolo_img_1, frcnn_img_1
    # return yolo_img_1.mul(yolo_hflip), frcnn_img_1.mul(frcnn_hflip)
    return yolo_img_0.mul(yolo_hflip), frcnn_img_0.mul(frcnn_hflip)


masks_path = 'masks'

if not os.path.exists(masks_path):
    os.makedirs(masks_path)

for img_name_index in range(len(files)):


    img_name = files[img_name_index]



    print()
    print(img_name_index,img_name)


    img_path0 = os.path.join('select1000_new', img_name)
    img_path1 = os.path.join('select1000_new_p', img_name)

    img0 = Image.open(img_path0).convert('RGB')
    img0_608 = resize_small(img0)
    boxes0_all = do_detect(darknet_model, img0_608, 0.5, 0.4, True)

    num_box = len(boxes0_all)
    print('Yolo detect:',num_box)

    result_p = inference_detector(mmdmodel,img_path0)

    result_above_confidence_num_ori = parse_mmd(result_p)
    print('RCNN detect:',result_above_confidence_num_ori)

    mmd_imgs,_ = inference_detector2(detector,img_path0)


    ori_imgs_t = resize2(Image.open('select1000_new/'+img_name).convert('RGB')).unsqueeze(0).cuda()

    img_mask_path = os.path.join(masks_path, img_name)

    # mask = read_mask(img_mask_path).to(device)

    num_std = 2.1
    flag = 0
    while not flag:
        num_std += 0.1
        mask,flag = get_mask_all_small(5000, ori_imgs_t, mmd_imgs, model=darknet_model,detector =detector,img_metas =img_metas,num_std=num_std)
    mask = mask.to(device)

    save_image(mask,os.path.join(masks_path, img_name))


    img0 = Image.open(img_path0).convert('RGB')
    img0_608 = resize_small(img0)

    width = img0_608.width
    height = img0_608.height
    img0 = torch.ByteTensor(torch.ByteStorage.from_buffer(img0_608.tobytes()))
    img0 = img0.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img0 = img0.view(1, 3, height, width)
    img0 = img0.float().div(255.0)
    img0 = img0.to(device)


    delta = torch.FloatTensor(1, 3, 500, 500).cuda()
    torch.nn.init.normal_(delta, mean=0, std=1.)
    delta.data = clamp(delta, 0. - ori_imgs_t, 1. - ori_imgs_t).mul_(mask)

    delta.requires_grad = True

    bestloss = 200000
    bestdalta = delta.data


    for p in range(800):

        # yolo_input, frcnn_input = pad_flip(img0, mmd_imgs,delta)

        list_boxes = darknet_model(img0+upyolo(delta))
        obj_confs_list = []
        for idx, box in enumerate(list_boxes):
            obj_confs_list.append(torch.cat((box[0][4].view(-1),box[0][4+85].view(-1),box[0][4+170].view(-1)),0))
        obj_confs_yolo = torch.cat([obj_confs_list[i] for i in range(len(obj_confs_list))],0)

        obj_conf_thresh_rcnn = 0.3
        obj_conf_thresh = 0.5

        obj_confs_yolo = sigm(obj_confs_yolo)

        result_above_confidence_num_yolo = len(obj_confs_yolo[obj_confs_yolo >0.45])
        

        obj_confs_yolo = obj_confs_yolo[obj_confs_yolo > obj_conf_thresh]

        obj_loss = 0

        frcnn_input = normalize(unnormalize(mmd_imgs)+uprcnn(delta))
        
        obj_confs_rcnn,bk_scores = getmm_list2(detector,frcnn_input,img_metas)

        result_above_confidence_num_rcnn = len(obj_confs_rcnn[obj_confs_rcnn >0.25])

        obj_confs_rcnn = obj_confs_rcnn[obj_confs_rcnn > 0.1]

        if result_above_confidence_num_rcnn==0:
            obj_confs_rcnn = []
        
        if result_above_confidence_num_yolo==0:
            obj_confs_yolo = []

        if (result_above_confidence_num_rcnn+result_above_confidence_num_yolo)<=bestloss:
            bestloss = result_above_confidence_num_rcnn+result_above_confidence_num_yolo
            bestdalta = delta.data

        if (len(obj_confs_yolo)!=0) or (len(bk_scores)!=0) :

            if len(obj_confs_yolo):
                targets_yolo = torch.ones_like(obj_confs_yolo).to(device)
                obj_loss += loss(obj_confs_yolo,targets_yolo)

            if len(bk_scores):
                targets_bk = torch.LongTensor([80 for i in range(len(bk_scores))]).to(device)
                obj_loss -=lossce(bk_scores,targets_bk)

            obj_loss.backward()
            grad = delta.grad.detach()
            d = alpha*torch.sign(grad)
            delta.data = clamp(delta+d, 0. - ori_imgs_t, 1. - ori_imgs_t).mul_(mask)
            delta.grad.zero_()

        else:
            with open("hnm_pgd.txt", "a") as output:
                output.write(str(img_name_index)+'-hnm_pgd: '+str(img_name)+' pgd break! in round '+str(p)+'\n')
            break

    save_image(ori_imgs_t+bestdalta,img_path1)

    img_new_608 = resize_small(Image.open(img_path1).convert('RGB'))

    boxes0 = do_detect(darknet_model, img_new_608, 0.5, 0.4, True)
    
    result_p = inference_detector(mmdmodel,img_path1)

    result_above_confidence_num_p = parse_mmd(result_p)

    print('hnm_pgd:',img_name,'done!, Yolo detect left',len(boxes0),', RCNN detect left',result_above_confidence_num_p)


    img0 = Image.open('select1000_new/'+img_name).convert('RGB')
    img1 = Image.open(img_path1).convert('RGB')
    img0_t = resize2(img0).cuda()
    img1_t = resize2(img1).cuda()
    img_minus_t = img0_t - img1_t

    unsatified,num_patch,total_area_rate,total_area = conntet_test(img_minus_t)
    print(unsatified,num_patch,total_area_rate,total_area)
    if unsatified:
        print(img_name, 'fail! >10 patch:',num_patch,'area:', total_area_rate)
        if img_name not in fail_image:
            fail_image.append(img_name)

    with open("hnm_pgd.txt", "a") as output:
        output.write(str(img_name_index)+'-fail_image: '+str(img_name)+', num_patch: '+str(num_patch)+'-'+str(total_area)+'!, Yolo left '+str(len(boxes0))+', RCNN left '+str(result_above_confidence_num_p)+'\n')

print(fail_image)




