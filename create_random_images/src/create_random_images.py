# coding: utf-8

import sys
sys.path.append('../')
import os
import settings
import numpy as np
import json
import pickle
import random

from PIL import Image
from scipy.misc import imread
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring


pi_names = settings.PI_NAMES
path_prefix = os.sep.join((
    settings.DATA_DIR,
    "images",
    "pi/"
))
n_pi_sampling = 14
pi_images = list()
dict_pi_images = dict()
for pi_name in pi_names:
    dict_pi_images[pi_name] = imread(path_prefix + pi_name + ".png")[:, :, :3]
        
bg_color = [95, 141, 154]
wmargin = 200

npi = len(pi_names) 
def create_random_images():
    npi_sample = np.random.randint(7, n_pi_sampling) # 画像に何牌おくかランダムで決定
    pi_labels = random.sample(dict_pi_images.keys(), npi_sample) # 何の牌を置くかランダムで決定
    shapes = [pi_image.shape for pi_image in dict_pi_images.values()] 
    max_shape = np.array(shapes).max(axis=0) # 全牌の最大shapeを取得 [h, w, c]
    img = np.zeros([max_shape[0], max_shape[1]*npi_sample, max_shape[2]],
                   dtype="uint8") # 牌の最大width、最大height*牌の数、チャンネル数の画像を生成
    img[:] = np.array(bg_color) # 背景で初期化 横長の画像ができるはず

    woffset = 0
    list_labels = list()
    list_bboxes = list()
    for pi_name in pi_labels:
        list_labels.append(pi_name)
        pi_image = dict_pi_images[pi_name]
        pi_shape = pi_image.shape # 牌のshape
        hoffset = int((max_shape[0] - pi_shape[0])*np.random.rand()) # (牌の最大width-牌のwidht)*random(0~1)
        xmin = woffset # xmin:固定値ゼロ
        xmax = woffset+pi_shape[1] # max:0+牌の幅
        ymin = hoffset # (牌の最大width-牌のwidht)*random(0~1)
        ymax = hoffset+pi_shape[0] # (牌の最大width-牌のwidht)*random(0~1) + 牌の高さ
        img[ymin:ymax, xmin:xmax, :] = pi_image # 上記で作成したBBOXの範囲に牌の画像に置き換える
        woffset = woffset + pi_shape[1] # 次の牌のxminは前の牌の横幅からスタート

        # BBOX座標の取得
        bboxes = np.array([xmin, ymin, xmax, ymax])
        list_bboxes.append(bboxes)
        
    max_pix = np.array(img.shape) + wmargin # 画像にmarginを足す ※チャンネルまで足されてるけどいいのかな？
    max_pix[0] = max_pix[1] # 正方形にする
    bg = np.zeros([max_pix[0], max_pix[1], max_shape[2]], dtype="uint8") # 上記で作った正方形のサイズの背景画像を生成
    bg[:] = np.array(bg_color)
    hoffset = np.random.randint(0, int(max_pix[0] - img.shape[0])) # 牌が配置された横長の画像をランダムに貼り付ける
    woffset = np.random.randint(0, wmargin) # 牌が配置された横長の画像をランダムに貼り付ける
    bg[hoffset:hoffset+img.shape[0], woffset:woffset+img.shape[1], :] = img # 牌が配置された横長の画像をランダムに貼り付ける
    img = bg 

    img_pix = list_bboxes
    img_pix[:] += np.array([woffset, hoffset, woffset, hoffset]) # BBOXの実数値

    return img, list_labels, list_bboxes

def save_image(img, fname):
    """画像(np.array)を.jpgで保存する"""
    pil_img = Image.fromarray(img) 
    imgKey = os.path.join(DIR_SAVE_IMAGE, fname + '.jpg')
    pil_img.save(imgKey)


def save_xml(fname, img_hight, img_width, img_channel,
             list_labels, list_bboxes):

    annotation = Element('annotation')

    foldername = SubElement(annotation, 'folder')
    foldername.text = DIR_SAVE_IMAGE

    filename = SubElement(annotation, 'filename')
    filename.text = fname + ".jpg"

    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_width)
    height = SubElement(size, 'height')
    height.text = str(img_hight)
    depth = SubElement(size, 'depth')
    depth.text = str(img_channel)

    for label_name, bbox in zip(list_labels, list_bboxes):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        _object = SubElement(annotation, 'object')
        name = SubElement(_object, 'name')
        name.text = label_name
        pose = SubElement(_object, 'pose')
        pose.text = 'front'
        truncated = SubElement(_object, 'truncated')
        truncated.text = "0"
        difficult = SubElement(_object, 'difficult')
        difficult.text = "0"
        occluded = SubElement(_object, 'occluded')
        occluded.text = "0"
        bndbox = SubElement(_object, 'bndbox')
        _xmin = SubElement(bndbox, 'xmin')
        _xmin.text = str(xmin)
        _ymin = SubElement(bndbox, 'ymin')
        _ymin.text = str(ymin)
        _xmax = SubElement(bndbox, 'xmax')
        _xmax.text = str(xmax)
        _ymax = SubElement(bndbox, 'ymax')
        _ymax.text = str(ymax)

        string = tostring(annotation, 'utf-8')
        pretty_string = minidom.parseString(string).toprettyxml(indent='    ')

    xml_file = os.path.join(DIR_SAVE_XML, fname + '.xml')
    with open(xml_file, 'w') as f:
        f.write(pretty_string)

DIR_SAVE_IMAGE = '/home/rio.kurihara/work/vision_kit/data/images/train'
DIR_SAVE_XML = '/home/rio.kurihara/work/vision_kit/data/annotations/xmls/'
DIR_SAVE_TXT = '/home/rio.kurihara/work/vision_kit/data/annotations/'

image_n = 10000
list_fname = []

for i in range(0, image_n):
    print(i)
    fname = 'image_' + str(i)
    list_fname.append(fname)
    img, list_labels, list_bboxes  = create_random_images() # ランダムに画像を生成して、BBOX情報を取得
    img_hight, img_width, img_channel = img.shape

    save_xml(fname, img_hight, img_width, img_channel,
                 list_labels, list_bboxes)
    save_image(img, fname) # 画像を保存

f = open(os.path.join(DIR_SAVE_TXT, 'trainval.txt'), 'w')
for x in list_fname:
    f.write(str(x) + "\n")
f.close()