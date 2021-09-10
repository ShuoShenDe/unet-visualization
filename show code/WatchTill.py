import os, sys
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import fastai
import fastai
from fastai.vision import *
from matplotlib.patches import Rectangle
import os
from PIL import Image

# path to image and label (both has to be TIF)

image_path = 'Y:/Users/Shuo/S1B_IW_GRDH_1SDV_20200802T114539_20200802T114613_022744_02B2A7_1C48/S1B_20200802T114539_1C48_compressed.tif'

with open('low_iou_file.txt') as f:
    lines = f.readlines()
    file_list = lines[0].replace("[", "").replace("]", "").replace("'", "").split(",")


# path to where image and labels will be tiled
tile_img = 'D:/tile_strategy/flood_mapping/'

# read the tiff file
image = cv.imread(image_path, -1)

# Remember to put a '_'after the name in order to get name_i_j
file_name = []
tile_size = (256, 256)
offset = (256, 256)

row = int(math.ceil(image.shape[0] / (offset[1] * 1.0)))
column = int(math.ceil(image.shape[1] / (offset[0] * 1.0)))

arr = os.listdir('Y:\\Users\\Shuo\\tiles_ST1_20200730_MMR_img')
plt.imshow(image)

for file in file_list:
    file_name = file.split(".")[0].split("_")
    if "ST1_20200808_MMR_1C48" not in file:
        continue
    #file_name = [78, 39]

    i = int(file_name[-1])
    j = int(file_name[-2])
    x1 = j * tile_size[0]
    y1 = i * tile_size[1]
    x2 = x1 + offset[0]
    y2 = y1 + offset[1]

    # Cropping an image
    # cropped_image = image[x1:x2, y1:y2]
    # plt.imshow(cropped_image)
    # plt.show()
    print(file_name)

    # plt.gca().add_patch(
    #     Rectangle((offset[0] * i, offset[1] * j), offset[0], offset[1], linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(
        Rectangle((tile_size[1] * i + offset[1], tile_size[0] * j + offset[0]), tile_size[1], tile_size[0], linewidth=1,
                  edgecolor='r', facecolor='none'))


plt.show()



# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#
# x1 = 48  # From save_name = 'ST1_20200808_MMR_1C48_i_j' this is j
# y1 = 13  # #From save_name = 'ST1_20200808_MMR_1C48_i_j' this is i
# x1 = x1 * tile_size[0]
# y1 = y1 * tile_size[1]
# x2 = x1 + offset[0]
# y2 = y1 + offset[1]
# image_tile_back = cv.imread(image_path, -1)  # read the tiff file
#
# # Make the zoom-in plot:
# fig = plt.figure(1, [15, 15])
# ax = fig.add_subplot(111)
# ax.imshow(image)
# axins = zoomed_inset_axes(ax, 20, loc=1)  # zoom =20
# axins.imshow(image_tile_back, interpolation="nearest", origin="lower")
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# plt.xticks(visible=False)
# plt.yticks(visible=False)
#
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.show()
