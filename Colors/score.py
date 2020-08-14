# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'


from scipy.io import loadmat
import csv
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image
import cv2
import json
import os



colors = loadmat('data/color150.mat')['colors']
unlikely_labels = ['sky', 'tree', 'road;route', 'grass', 'sidewalk;pavement', 'earth;ground', 'mountain;mount',
'car;auto;automobile;machine;motorcar', 'house', 'sea', 'field', 'fence;fencing', 'rock;stone', 'sand', 'skyscraper', 
'grandstand;covered;stand', 'path', 'runway', 'river', 'bridge;span', 'flower', 'hill', 'palm;palm;tree','boat', 'arcade;machine', 
'hovel;hut;hutch;shack;shanty', 'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle', 
'tower', 'streetlight;street;lamp', 'booth;cubicle;stall;kiosk','airplane;aeroplane;plane', 'dirt;track', 'pole', 'land;ground;soil', 
'stage', 'van', 'ship', 'fountain', 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter', 'canopy', 
'washer;automatic;washer;washing;machine', 'swimming;pool;swimming;bath;natatorium', 'plaything;toy', 'barrel;cask', 
'basket;handbasket', 'waterfall;falls', 'tent;collapsible;shelter', 'minibike;motorbike', 'cradle', 'oven', 'ball', 
'tank;storage;tank', 'trade;name;brand;name;brand;marque', 'animal;animate;being;beast;brute;creature;fauna', 
'bicycle;bike;wheel;cycle', 'lake', 'screen;silver;screen;projection;screen', 'sculpture', 'hood;exhaust;hood', 
'traffic;light;traffic;signal;stoplight', 'pier;wharf;wharfage;dock', 'crt;screen', 'plate', 'bulletin;board;notice;board']



from PIL import Image



id_colors = []
color_dict = {}

with open('data/object150_info.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for idx, row in enumerate(reader):
        row.append(colors[idx])
        del row[1:5]
        row[0] = int(row[0])
        id_colors.append(row)
    
color_dict = {tuple(c[2]): c[1] for c in id_colors }



def crop(image):
    image = cv2.imread(image)
    height, width = image.shape[:2]

    # Let's get the starting pixel coordiantes (top left of cropped top)
    start_row, start_col = int(0), int(width * .5)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_col, end_row = int(width), int(height)
    cropped_top = image[start_row:end_row , start_col:end_col]

    # imshow(cropped_top)
    pixels = height * (width/2)
    return cropped_top, pixels



result = {}
imgs = {}
sum_pix = 0
c_pix = 0
for img in os.listdir('Results'):
    path = os.path.join('Results', img)
    img, px = crop(path)
    unrealistic = 0  
    for row in img:
        for col in row:
            if (tuple(col) in color_dict) and color_dict[tuple(col)] in unlikely_labels:
                unrealistic = unrealistic + 1
    # percent realistic is total - unrealistic over / total
    # realistic is total - unrealistic
    p = (px-unrealistic)/px
    imgs[path] = p * 100

    # avg percent realistic
    c_pix += int(px * p)
    sum_pix += px

    
result["Num Images"] = 1000
result["Dataset AVG"] = (c_pix / sum_pix) * 100
result["images"] = imgs
    
with open('data.json', 'w') as outfile:
    json.dump(result, outfile)        
            





