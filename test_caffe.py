import caffe
import numpy as np
import cv2
import os


net_file = 'classifier.prototxt'
caffe_model = 'classifier.caffemodel'

net = caffe.Net(net_file, caffe_model, caffe.TEST)

filelist = ''
for file in filelist:
    if not (file.endswith('.jpg') or file.endswith('.png')):
        continue
    img = cv2.imread(file)
    if img is None:
        continue
    img = cv2.resize(img, (96, 96))
    input = preprocess(img)
    net.blobs['blob0'].reshape(1, 3, 96, 96)
    net.blobs['blob0'].data[...] = input
    print('blob0:', net.blobs['blob0'].data)
    out = net.forward()['softmax_blob114']
    label = np.argmax(out)
