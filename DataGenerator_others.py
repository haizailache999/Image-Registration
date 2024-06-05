__author__ = 'Yifeng Wang'

import torch
from torch.utils.data import Dataset
#import pandas as pd
import numpy as np
from os.path import join
from copy import copy
import math
import numpy as np
import torch
from os.path import join
import random
from dataAugmentation import MRIDataAugmentation
import scipy.ndimage
import scipy.linalg as linalg
#from medmnist import VesselMNIST3D as MNIST
#from medmnist import FractureMNIST3D as MNIST
from medmnist import NoduleMNIST3D as MNIST
#from medmnist import OrganMNIST3D as MNIST        #Please choose which dataset you want

def sphere(shape, radius, position):
    semisizes = (radius,) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
     arr += (np.abs(x_i/semisize) ** 2)
    return arr <= 1.0


def loc_convert(loc, axis, radian):
    radian = np.deg2rad(radian)
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    new_loc = np.dot(rot_matrix, loc)
    return new_loc

def extract_slice(img, c, v, radius):
    epsilon = 1e-12
    x = np.arange(-radius, radius, 1)
    y = np.arange(-radius, radius, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    loc = np.array([X.flatten(), Y.flatten(), Z.flatten()])

    hspInitialVector = np.array([0, 0, 1])
    h_norm = np.linalg.norm(hspInitialVector)
    h_v = hspInitialVector / h_norm
    h_v[h_v == 0] = epsilon
    v = v / np.linalg.norm(v)
    v[v == 0] = epsilon

    hspVecXvec = np.cross(h_v, v) / np.linalg.norm(np.cross(h_v, v))
    acosineVal = np.arccos(np.dot(h_v, v))
    hspVecXvec[np.isnan(hspVecXvec)] = epsilon
    acosineVal = epsilon if np.isnan(acosineVal) else acosineVal

    loc = loc_convert(loc, hspVecXvec, 180 * acosineVal / math.pi)
    sub_loc = loc + np.reshape(c, (3, 1))
    loc = np.round(sub_loc)
    loc = np.reshape(loc, (3, X.shape[0], X.shape[1]))

    sliceInd = np.zeros_like(X, dtype=float)
    slicer = np.copy(sliceInd)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if loc[0, i, j] >= 0 and loc[0, i, j] < img.shape[0] and loc[1, i, j] >= 0 and loc[1, i, j] < img.shape[1] and loc[2, i, j] >= 0 and loc[2, i, j] < img.shape[2]:
                slicer[i, j] = img[
                    loc[0, i, j].astype(int), loc[1, i, j].astype(int), loc[2, i, j].astype(int)]
    return slicer, sub_loc, loc

def is_point_in_block(point, block_min, block_max):
    p=point
    min_val=block_min
    max_val=block_max
    if ((min_val[0]<=p[0][0]<=max_val[0] and min_val[1]<=p[0][1]<=max_val[1] and min_val[2]<=p[0][2]<=max_val[2]) and
    (min_val[0]<=p[1][0]<=max_val[0] and min_val[1]<=p[1][1]<=max_val[1] and min_val[2]<=p[1][2]<=max_val[2]) and
    (min_val[0]<=p[2][0]<=max_val[0] and min_val[1]<=p[2][1]<=max_val[1] and min_val[2]<=p[2][2]<=max_val[2]) and 
    (min_val[0]<=p[3][0]<=max_val[0] and min_val[1]<=p[3][1]<=max_val[1] and min_val[2]<=p[3][2]<=max_val[2])):
        return True
    return False

#The next two functions are to control the segmentation for the blocks and labels
def getposition_1(check):
    final_list=[]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                block_min_coords = (i*5, j*5, k*5)
                block_max_coords = (i*5+9, j*5+9, k*5+9)
                checkin=is_point_in_block(check,block_min_coords,block_max_coords)
                if checkin==True:
                    final_list.append(i*9+j*3+k*1)
    return final_list
                
def getposition_2(block_min_coord,check):
    final_list=[]
    origin_min_coords=block_min_coord
    for i in range(3):
        for j in range(3):
            for k in range(3):
                block_min_coords = (origin_min_coords[0]+i*2,origin_min_coords[1]+j*2,origin_min_coords[2]+k*2)
                block_max_coords = (block_min_coords[0]+5, block_min_coords[1]+5, block_min_coords[2]+5)
                checkin=is_point_in_block(check,block_min_coords,block_max_coords)
                if checkin==True:
                    final_list.append(i*9+j*3+k*1)
    return final_list

class MRIDataGenerator(Dataset):

    def __init__(self, img_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 batchSize=16,
                 dim=(20, 20, 20),          #Size of 3D volume
                 n_channels=1,              #gray image should have one channel
                 n_classes=2,
                 augmented=False,
                 augmented_fancy=False,
                 MCI_included=True,
                 MCI_included_as_soft_label=False,
                 returnSubjectID=False,
                 dropBlock = False,
                 dropBlockIterationStart = 0,
                 gradientGuidedDropBlock=False
                 ):
        # 'Initialization'

        self.img_dir = ''
        self.split = split
        self.transform = transform
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.batch_size = batchSize
        self.dim = dim
        self.dim2d=(4,4)                         #This is to control the size of 2D images
        self.dimlabel1=(27,)                     #This is to control the label number (number of blocks)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augmented = augmented
        self.augmented_fancy = augmented_fancy
        self.MCI_included = MCI_included
        self.MCI_included_as_soft_label = MCI_included_as_soft_label
        self.returnSubjectID = returnSubjectID
        self.dropBlock = dropBlock
        self.train_dataset = MNIST(split="train", download=True,size=28)
        self.val_dataset = MNIST(split="val", download=False,size=28)
        self.test_dataset = MNIST(split="test", download=False,size=28)
        self.dropBlock_iterationCount = dropBlockIterationStart
        self.gradientGuidedDropBlock = gradientGuidedDropBlock

        self.parse_csv_file2()
        self.on_epoch_end()

        self.dataAugmentation = MRIDataAugmentation(self.dim, 0.5)

    def __len__(self):
        self.on_epoch_end()
        return math.ceil(self.totalLength/self.batch_size)

    def combine(self,image,batchSize):
        imaging=image.squeeze(dim=4)
        return imaging
    
    def __getitem__(self, idx):
        if self.split == 'train':
            if not self.returnSubjectID:  
                images_3d, images_2d_list = self._load_batch_image_train(idx)
                images = images_3d.astype(np.float32)
                images=torch.from_numpy(images)
                images=self.combine(images,self.batch_size)
                image_2d=np.zeros((self.batch_size, *self.dim2d))
                labels1_loss1=np.zeros((self.batch_size,*self.dimlabel1),dtype=np.int64)
                labels2_loss=[]
                labels2=[]
                labels1=[]
                for i in range(self.batch_size):
                    image_single=images[i:i+1,:,:,:]
                    c=images_2d_list[i]
                    n=[random.randint(0, 9),random.randint(0, 9),random.randint(0, 9)]
                    r=2                   #This is for 2D image size. 2 means the image is 4*4. If you want an image with 2*2, then use r=1.
                    arr=torch.squeeze(image_single)
                    slicer, sub_loc, slice_check = extract_slice(arr, c, n, r)
                    check_point1=(slice_check[0][0][0],slice_check[1][0][0],slice_check[2][0][0])
                    check_point2=(slice_check[0][2*r-1][0],slice_check[1][2*r-1][0],slice_check[2][2*r-1][0])
                    check_point3=(slice_check[0][0][2*r-1],slice_check[1][0][2*r-1],slice_check[2][0][2*r-1])
                    check_point4=(slice_check[0][2*r-1][2*r-1],slice_check[1][2*r-1][2*r-1],slice_check[2][2*r-1][2*r-1])
                    check=[check_point1,check_point2,check_point3,check_point4]
                    label_list=getposition_1(check)
                    image_2d[i,:,:]=slicer
                    final_multi_label1=np.zeros(27)            #Total 27 blocks, so label number is 27
                    for label_number in label_list:
                        final_multi_label1[label_number]=1
                    labels1_loss1[i,:] = final_multi_label1
                    labels1.append(label_list)
                    labels2_loss_mid=[]
                    labels2_mid=[]
                    for i_2 in range(len(label_list)):
                        a=label_list[i_2]//9                          #This is to calculate the block coordinate. 27=3*3*3
                        b=(label_list[i_2]-a*9)//3
                        c=label_list[i_2]-a*9-b*3
                        min_cord_2=[a*5,b*5,c*5]
                        label_list_2=getposition_2(min_cord_2,check)
                        final_multi_label_2=np.zeros(3*3*3)
                        for label_number in label_list_2:
                            final_multi_label_2[label_number]=1
                        labels2_loss_mid.append(final_multi_label_2)
                        labels2_mid.append(label_list_2)
                    labels2_loss.append(labels2_loss_mid)
                    labels2.append(labels2_mid)
                labels1_loss1=torch.from_numpy(labels1_loss1)
                return images_3d, image_2d, labels1,labels1_loss1,labels2,labels2_loss
        else:
            if self.split=='test':
                images_3d, images_2d_list = self._load_batch_image_test(idx)
            else:
                images_3d, images_2d_list = self._load_batch_image_val(idx)
            images = images_3d.astype(np.float32)
            images=torch.from_numpy(images)
            images=self.combine(images,self.batch_size)
            image_2d=np.zeros((self.batch_size, *self.dim2d))
            labels1_loss1=np.zeros((self.batch_size,*self.dimlabel1),dtype=np.int64)
            labels2_loss=[]
            labels2=[]
            labels1=[]
            for i in range(self.batch_size):
                image_single=images[i:i+1,:,:,:]
                c=images_2d_list[i]
                n=[random.randint(0, 9),random.randint(0, 9),random.randint(0, 9)]
                r=2
                arr=torch.squeeze(image_single)
                slicer, sub_loc, slice_check = extract_slice(arr, c, n, r)
                check_point1=(slice_check[0][0][0],slice_check[1][0][0],slice_check[2][0][0])
                check_point2=(slice_check[0][2*r-1][0],slice_check[1][2*r-1][0],slice_check[2][2*r-1][0])
                check_point3=(slice_check[0][0][2*r-1],slice_check[1][0][2*r-1],slice_check[2][0][2*r-1])
                check_point4=(slice_check[0][2*r-1][2*r-1],slice_check[1][2*r-1][2*r-1],slice_check[2][2*r-1][2*r-1])
                check=[check_point1,check_point2,check_point3,check_point4]
                label_list=getposition_1(check)
                image_2d[i,:,:]=slicer
                final_multi_label1=np.zeros(27)
                for label_number in label_list:
                    final_multi_label1[label_number]=1
                labels1_loss1[i,:] = final_multi_label1
                labels1.append(label_list)
                labels2_loss_mid=[]
                labels2_mid=[]
                for i_2 in range(len(label_list)):
                    a=label_list[i_2]//9
                    b=(label_list[i_2]-a*9)//3
                    c=label_list[i_2]-a*9-b*3
                    min_cord_2=[a*5,b*5,c*5]
                    label_list_2=getposition_2(min_cord_2,check)
                    final_multi_label_2=np.zeros(3*3*3)
                    for label_number in label_list_2:
                        final_multi_label_2[label_number]=1
                    labels2_loss_mid.append(final_multi_label_2)
                    labels2_mid.append(label_list_2)
                labels2_loss.append(labels2_loss_mid)
                labels2.append(labels2_mid)
            labels1_loss1=torch.from_numpy(labels1_loss1)
            return images_3d, image_2d, labels1,labels1_loss1,labels2,labels2_loss

    def parse_csv_file2(self):
        self.file_path_train=[]
        self.file_path_val=[]
        self.file_path_test=[]
        random.seed(3407)
        train_big_block=range(0,len(self.train_dataset))
        val_big_block=range(0,len(self.val_dataset))
        test_big_block=range(0,len(self.test_dataset))
        train_small_piece=random.sample(range(0,17*17),50)
        val_small_piece=random.sample(range(0,17*17),50)
        test_small_piece=random.sample(range(0,17*17),50)
        i_6_list=random.sample(range(0,20),4)
        for i in train_big_block:
            for t in train_small_piece:
                i_4=t//17
                i_5=t%17
                if 17>=i_4>=2 and 17>=i_5>=2:
                    for i_6 in i_6_list:
                        self.file_path_train.append([i,i_4,i_5,i_6])
        for i in val_big_block:
            for t in val_small_piece:
                i_4=t//17
                i_5=t%17
                if 17>=i_4>=2 and 17>=i_5>=2:
                    for i_6 in i_6_list:
                        self.file_path_val.append([i,i_4,i_5,i_6])
        for i in test_big_block:
            for t in test_small_piece:
                i_4=t//17
                i_5=t%17
                if 17>=i_4>=2 and 17>=i_5>=2:
                    for i_6 in i_6_list:
                        self.file_path_test.append([i,i_4,i_5,i_6])
        if self.split == 'train':
            self.totalLength = len(self.file_path_train)
        elif self.split=='val':
            self.totalLength = len(self.file_path_val)
        else:
            self.totalLength = len(self.file_path_test)
        print(self.split,self.totalLength)

    def on_epoch_end(self):
        if self.split == 'train':
            np.random.shuffle(self.file_path_train)

    def _load_one_image(self, image_path,dataset):
        image_MRI=dataset[image_path[0]][0][0]
        initial_shape=image_MRI.shape
        image_MRI=scipy.ndimage.zoom(image_MRI, [20/initial_shape[0],20/initial_shape[1],20/initial_shape[2]], order=3)
        final_3d=np.expand_dims(image_MRI, axis=0)
        return final_3d

    def _rotate_idx(self, l, m):
        for i in range(len(l)):
            while l[i] >= m:
                l[i] = l[i] - m
        return l

    def _load_batch_image_train(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_train))
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list=[]
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.file_path_train[idxlist[i]],self.train_dataset)
            images_2d_list.append([self.file_path_train[idxlist[i]][1],self.file_path_train[idxlist[i]][2],self.file_path_train[idxlist[i]][3]])
        return images_3d,images_2d_list

    def _load_batch_image_test(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_test))
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list=[]
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.file_path_test[idxlist[i]],self.test_dataset)
            images_2d_list.append([self.file_path_test[idxlist[i]][1],self.file_path_test[idxlist[i]][2],self.file_path_test[idxlist[i]][3]])
        return images_3d,images_2d_list

    def _load_batch_image_val(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_val))
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list=[]
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.file_path_val[idxlist[i]],self.val_dataset)
            images_2d_list.append([self.file_path_val[idxlist[i]][1],self.file_path_val[idxlist[i]][2],self.file_path_val[idxlist[i]][3]])
        return images_3d,images_2d_list

