import os, sys
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

import fastai
import fastai
from fastai.vision import *

image_path = 'Y:/Users/Shuo/S1B_IW_GRDH_1SDV_20200802T114539_20200802T114613_022744_02B2A7_1C48/S1B_20200802T114539_1C48_compressed.tif'
tile_img = 'D:/image_tile/'
image = cv.imread(image_path, -1)  # read the tiff file

save_name = 'ST1_20200808_MMR_1C48_'  # Remember to finish with "_"
tile_size = (256, 256)
offset = (256, 256)

print("Tiling: started...")

flag = 0  # flag value that corresponds to the pixel value in the frame
count = 0
arr = os.listdir('Y:\\Users\\Shuo\\tiles_ST1_20200730_MMR_img')
for file in arr:
    file_name = file.split(".")[0].split("_")
    print(file_name)
    i = int(file_name[-1])
    j = int(file_name[-2])
    cropped_img = image[offset[1] * i:min(offset[1] * i + tile_size[1], image.shape[0]),
                  offset[0] * j:min(offset[0] * j + tile_size[0], image.shape[1])]
    cv.imwrite(tile_img + save_name + str(i) + "_" + str(j) + ".png", cropped_img)

# for i in tqdm(range(int(math.ceil(image.shape[0] / (offset[1] * 1.0))))):
#     for j in range(int(math.ceil(image.shape[1] / (offset[0] * 1.0)))):
#
#         cropped_img = image[offset[1] * i:min(offset[1] * i + tile_size[1], image.shape[0]),
#                       offset[0] * j:min(offset[0] * j + tile_size[0], image.shape[1])]
#         # exlude tiles if one flag-pixel is in the tile and tiles outside the AnalysisExtent
#
#         # exlude tiles if one flag-pixel is in the tile and tiles outside the AnalysisExtent and tiles without flood-pixels (optional)
#         # if np.sum(cropped_img==flag) == 0 and np.sum(cropped_AnalysisExtent)== tile_size[0]*tile_size[1] and np.sum(cropped_lab)>0:
#         count = count + 1
#
#         cv.imwrite(tile_img + save_name + str(i) + "_" + str(j) + ".png", cropped_img)


# file_name = [12, 54]
# i = int(file_name[-1])
# j = int(file_name[-2])
#
# x1 = j * tile_size[0]
# y1 = i * tile_size[1]
# x2 = x1 + offset[0]
# y2 = y1 + offset[1]
# cropped_image = image[x1:x2, y1:y2]
# plt.imshow(cropped_image)
# plt.show()
