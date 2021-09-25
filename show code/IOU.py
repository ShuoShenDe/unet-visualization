import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import numpy as np
import torch
import json
from collections import OrderedDict
import random
from PIL import Image
import matplotlib
import pandas as pd


def binaryMaskIOU(mask1, mask2):  # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    if (mask1_area + mask2_area - intersection != 0):
        iou = intersection / (mask1_area + mask2_area - intersection)
    else:
        iou = 1
    return iou


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pathX = 'Y:/Users/Shuo/tiles_ST1_20200730_MMR_img/'
    pathY = 'Y:/Users/Shuo/tiles_ST1_20200730_MMR_label/'
    pathY_hat = "D:/cern/model1/"
    base_path = "D:/unet-vis/"
    image_fold = "newimages"

    image_series = "ST1_20200808_MMR_1C48"
    predict_threshold = 0.8
    iou_threshold = 0.5

    filesX = os.listdir(pathX)

    mydata = []
    mycoords = []
    low_iou_file_name = []
    low_iou = []

    # initialization with points1.js
    with open('../data/points1.js') as dataFile:
        data = dataFile.read()
        obj = data[data.find('['): data.rfind(']') + 1]
        jsonObj = json.loads(obj)
        for i in jsonObj:
            pointIOU = i["iou"]
            pointCoord = i["coords"]
            mycoords.append([pointIOU, pointCoord])
    mycoords = sorted(mycoords, key=lambda x: x[0])


    for f in filesX:
        if image_series not in f:
            continue
        im = imageio.imread(pathX + f)
        pred = np.load(pathY_hat + f.split('.')[0] + '.npy')
        pred[pred >= predict_threshold] = 1
        pred[pred < predict_threshold] = 0
        imY = imageio.imread(pathY + f)

        # save X, Y, Y_hat images to corresponding folders
        # matplotlib.image.imsave(base_path + image_fold +"/X/" + f, im)
        # matplotlib.image.imsave(base_path + image_fold + "/Y/"+f, imY)
        # matplotlib.image.imsave(base_path + image_fold +"/Y_hat/" + f, pred)

        iou = binaryMaskIOU(pred, imY)
        if iou < iou_threshold:
            low_iou_file_name.append(f)
            low_iou.append(iou)
        nearestPoint = min(mycoords, key=lambda x: abs(x[0] - iou))[0]
        coordslist = [x for x in mycoords if x[0] == nearestPoint]

        # save info in points.js
        data = {"mask_path_web": image_fold + "/Y/" + f, "iou": iou,
                "pred_path": "http://0.0.0.0:8000/" + image_fold + "/Y_hat/" + f,
                "coords": random.choice(coordslist)[1], "patch_path": "http://0.0.0.0:8000/" + image_fold + "/X/" + f,
                "patch_path_web": image_fold + "/X/" + f,
                "mask_path": "http://0.0.0.0:8000/" + image_fold + "/Y/" + f, "id": f.split('.')[0],
                "pred_path_web": image_fold + "/Y_hat/" + f}
        mydata.append(data)

    # save file name with low iou in csv
    pd = pd.DataFrame(low_iou_file_name, columns=['file_name'])
    pd['iou'] = low_iou
    pd.to_csv('file_name_with_low_iou.csv')

    # save file points.js
    print(mydata, file=open('points.js', 'w'))
    #s ave file low_iou_file_predict_threshold_
    print(low_iou_file_name, file=open(
        'low_iou_file_predict_threshold_' + predict_threshold.__str__() + 'iou_threshold' + iou_threshold.__str__() + '.txt',
        'w'))
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
