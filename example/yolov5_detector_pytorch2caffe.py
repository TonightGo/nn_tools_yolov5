import sys
sys.path.insert(0,'.')
sys.path.append('yolov5')
import torch
from torch.autograd import Variable
import pytorch_to_caffe
import numpy as np

from models.yolo import Model
import yaml


if __name__=='__main__':
    name = 'yolov5'
    weight_path = 'best.pt'
    save_caffe_prototxt_path = 'best.prototxt'
    save_caffemodel_path = 'best.caffemodel'

    hyp = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    ckpt = torch.load(weight_path, map_location='cpu')
    model = Model(ckpt['model'].yaml, ch=3, nc=5, anchors=hyp.get('anchors'))
    model.load_state_dict(ckpt['model'].state_dict())
    model.eval()
    
    input=Variable(torch.ones([1,3,640,640]))
    pytorch_to_caffe.trans_net(model,input,name)
    pytorch_to_caffe.save_prototxt(save_caffe_prototxt_path)
    pytorch_to_caffe.save_caffemodel(save_caffemodel_path)
    # pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    # pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    
    