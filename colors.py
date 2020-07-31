from pycocotools.coco import COCO
import numpy as np


def getColors():
    coco = COCO("./HotelsCocos/hotels.json")



    ann_ids = coco.getAnnIds()
    ann_ids = coco.loadAnns(ids=ann_ids)
    catIds = list(set([i['category_id'] for i in ann_ids]))
    categories = coco.loadCats(ids=catIds)



    colors = []
    for cat in categories:
        h = cat['color'].lstrip("#")
        rgb = list(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
        colors.append(rgb, cat['name'])



    return np.array(colors)





