import csv
from pycocotools.coco import COCO
import numpy as np
from skimage import io
from glob import glob
import os
from tqdm import tqdm

def get_cats():
    coco = COCO("HotelsCocos/hotels.json")
    cats = coco.loadCats(coco.getCatIds())
    return cats

def get_ade_colors():
    colors = []

    with open('data/object150_info.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for idx, row in enumerate(reader):
            colors.append(row[-1])
    
    return colors

def get_hotels_colors():
    cats = get_cats()
    colors = []
    for idx, cat in enumerate(cats):
        hex = cat['color'].lstrip('#')
        rgb = list(tuple(int(hex[i:i+2], 16) for i in (0, 2, 4)))
        colors.append(str(cat['name']))
    return colors

def find(search_term, ade):
    search_term = search_term.replace(" ", "").split("/")

    for s in search_term:
        for a in ade:
            if s == "art":
                return -1
            if s in a:
                return ade.index(a) + 1
    return -1

def match(hotels, ade):
    new_colors = {}
    ex = 1
    for i, h in enumerate(hotels):
        idx = find(h, ade)
        if idx > 0:
            new_colors[i+1] = idx+1
        else:
            new_colors[i+1] = len(ade)  + ex
            ex+= 1
    return new_colors

def fix_pictures(new_colors):
    imgs = [f for f in glob("/pless_nfs/home/mdt_/semantic-segmentation-pytorch/imgs/HotelsData/*.png")]
    
    for img in tqdm(imgs, desc='Adjusting Hotels'):
        op = io.imread(img)
        for nc in new_colors.keys():
            op[op == nc] = new_colors[nc]
        io.imsave(os.path.join("imgs/Hotels150", os.path.basename(img)), op)

def new_colors_file(x):
    l = [i for i in range(1,x+1)]

    import pickle

    with open("adjusted_colors.pckl", "wb") as pckl:
        pickle.dump(l, pckl)
def main(args):
    if args.a:
        ade_colors = get_ade_colors()
        hotels_colors = get_hotels_colors()
        new_colors = match(hotels_colors, ade_colors)
        fix_pictures(new_colors)
    elif args.c:
        print("Creating color file...")
        new_colors_file(max(new_colors.values()))
    else:
        print("invalid")

def odgt():
    hotels_imgs = [f for f in glob("/pless_nfs/home/mdt_/semantic-segmentation-pytorch/HotelsAdjusted/*.png")]
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument("-a", help="adjust hotels to match ADE labels", action='store_true')
    # parser.add_argument("-c", help="create colors", action='store_true')
    # parser.add_argument("-o", help="create odgt files", action='store_true')

    args = parser.parse_args()

    if args.o:
        print("ODGT not yet implemented")
    main(args)