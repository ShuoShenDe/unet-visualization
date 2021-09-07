
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import numpy as np
import torch
import torchvision.ops.boxes as bops
import json
from collections import OrderedDict
import random
from PIL import Image
import matplotlib


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

    filesX = os.listdir(pathX)
    mydata = []
    mycoords = []
    file_name = []
    with open('D:/unet-vis/data/points1.js') as dataFile:
        data = dataFile.read()
        obj = data[data.find('['): data.rfind(']') + 1]
        jsonObj = json.loads(obj)
        for i in jsonObj:
            pointIOU = i["iou"]
            pointCoord = i["coords"]
            mycoords.append([pointIOU, pointCoord])
    mycoords = sorted(mycoords, key=lambda x: x[0])

    for f in filesX:
        im = imageio.imread(pathX + f)
        pred = np.load(pathY_hat + f.split('.')[0] + '.npy')
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        imageY = imageio.imread(pathY + f)

        # matplotlib.image.imsave(base_path + image_fold +"/X/" + f, f)
        # matplotlib.image.imsave(base_path + image_fold +"/Y_hat/" + f, pred)
        # matplotlib.image.imsave(base_path + image_fold + "/Y/"+f, imageY)

        iou = binaryMaskIOU(pred, imageY)
        if iou < 0.5:
            file_name.append(f)
            print(f)
            print(iou)
        nearestPoint = min(mycoords, key=lambda x: abs(x[0] - iou))[0]
        coordslist = [x for x in mycoords if x[0] == nearestPoint]
        # print(coordslist)
        data = {"mask_path_web": image_fold + "/Y/" + f, "iou": iou, "pred_path": "http://0.0.0.0:8000/"+image_fold+"/Y_hat/" + f,
                "coords": random.choice(coordslist)[1], "patch_path": "http://0.0.0.0:8000/"+image_fold+"/X/" + f,
                "patch_path_web": image_fold + "/X/" + f,
                "mask_path": "http://0.0.0.0:8000/"+image_fold+"/Y/"+f, "id": f.split('.')[0],
                "pred_path_web": image_fold + "/Y_hat/" + f}
        mydata.append(data)
    print(mydata, file=open('points.js', 'w'))
    print(file_name, file=open('low_iou_file.txt', 'w'))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
