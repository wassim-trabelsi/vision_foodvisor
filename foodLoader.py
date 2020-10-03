import torch
from torch.utils.data import Dataset
import os
import json
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from config import TomateSolide

class FoodLoader(Dataset):

    def __init__(self, root='data', transform=None, mode=None):

        self.root = root
        self.transform = transform
        self.folderpath = os.path.join(root, 'assignment_imgs')

        # Load images
        self.imgs = GetFilesFromDirectory(self.folderpath)
        if mode:
            l = len(self.imgs)
            sep1 = l * 2 // 3
            sep2 = l * 5 // 6
            if mode == 'train':
                self.imgs = self.imgs[:sep1]
            if mode == 'val':
                self.imgs = self.imgs[sep1:sep2]
            if mode == 'test':
                self.imgs = self.imgs[sep2:]

        # Load annotations
        annspath = os.path.join(root, 'img_annotations.json')
        with open(annspath) as f:
            self.anns = json.load(f, encoding='utf8')

        # Load mapping
        mappingpath = os.path.join(root, 'label_mapping.csv')
        with open(mappingpath, encoding="utf8") as infile:
            reader = csv.reader(infile)
            self.id2food = {rows[0]: rows[1] for rows in reader}
        with open(mappingpath, encoding="utf8") as infile:
            reader = csv.reader(infile)
            self.id2fooden = {rows[0]: rows[2] for rows in reader}

    def showanns(self, imagename=None, imageid=None):
        if imageid:
            imagename = self.imgs[imageid]
        elif imagename:
            imagename = imagename
        else:
            imagename = random.choice(self.imgs)
        img_path = os.path.join(self.folderpath, imagename)
        objects = self.anns[imagename]
        plt.figure(figsize=(15, 20))
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 2
        for obj in objects:
            if obj['is_background']: continue
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = create_polygon(obj['box'])
            polygons.append(poly)
            color.append(c)
            circle = Circle((obj['box'][0], obj['box'][1]), r)
            classid = obj['id']
            plt.text(obj['box'][0] + 2, obj['box'][1] - 5, self.id2food[classid], bbox=dict(facecolor=c, alpha=0.7))
            circles.append(circle)

        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)

    def __getitem__(self, idx):

        # load images and label

        imagename = self.imgs[idx]
        img_path = os.path.join(self.folderpath, imagename)
        img = Image.open(img_path).convert("RGB")
        has_tomate = torch.tensor(0)
        objects = self.anns[imagename]

        # Naive labeling.
        for obj in objects:
            classid = obj['id']
            food = self.id2food[classid]
            if food in TomateSolide:
                has_tomate = torch.tensor(1)
                break

        if self.transform:
            img = self.transform(img)

        return img, has_tomate, imagename

    def __len__(self):
        return len(self.imgs)

def GetFilesFromDirectory(directory):
    allfiles = []
    imids = []
    for root,dirs,files in os.walk(directory):
        for filespath in files:
            full_filepath = os.path.join(root, filespath)
            allfiles.append(full_filepath)
            imids.append(filespath)
        return imids

def create_polygon(box):
    xA,yA,w,h = box
    A = (xA,yA)
    B = (xA+w,yA)
    C = (xA+w, yA+h)
    D = (xA, yA+h)
    poly = Polygon([A,B,C,D])
    return poly