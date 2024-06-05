__author__ = 'Yifeng Wang'

#import cv2
#from PIL import Image
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
                 dim=(20, 20, 20),
                 n_channels=1,
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
        random.seed( 3407 )
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.batch_size = batchSize
        self.dim = dim
        self.dim2d=(4,4)
        self.dimlabel1=(27,)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augmented = augmented
        self.augmented_fancy = augmented_fancy
        self.MCI_included = MCI_included
        self.MCI_included_as_soft_label = MCI_included_as_soft_label
        self.returnSubjectID = returnSubjectID
        self.dropBlock = dropBlock
        d=self.create_MRI()
        self.imaged = (d - np.min(d)) / (np.max(d) - np.min(d))
        self.dropBlock_iterationCount = dropBlockIterationStart
        self.gradientGuidedDropBlock = gradientGuidedDropBlock
        self.parse_csv_file()
        self.on_epoch_end()

        self.dataAugmentation = MRIDataAugmentation(self.dim, 0.5)

    def create_MRI(self):
        d=np.zeros((1,200,200,200), dtype = int)
        path=[]
        csv_path = join(self.img_dir, f'split.pretrained.0.csv')
        text = [line.strip() for line in open(csv_path)]
        for line in text[1:]:
            items = line.split(',')
            if items[-2] == 'CN' or items[-2] == 'AD' or items[-2] == 'MCI':         
                image_path = join(self.img_dir, 'subjects', items[0], items[1], 'deeplearning_prepare_data',
                                  'image_based',
                                  't1_linear',
                                  items[0] + '_' + items[
                                      1] + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')
                path.append(image_path)
                if len(path)==10*10*10:
                    break
        for i in range(len(path)):
            single_path=path[i]
            image_MRI=torch.load(single_path)
            image_MRI=image_MRI[0]
            initial_shape=image_MRI.shape
            image_MRI=scipy.ndimage.zoom(image_MRI, [20/initial_shape[0],20/initial_shape[1],20/initial_shape[2]], order=3)         #Here we zoom in the 3D data into 20*20*20
            z=i//100
            y=(i-z*100)//10
            x=i-z*100-10*y
            d[0,x*20:(x+1)*20,y*20:(y+1)*20,z*20:(z+1)*20]=image_MRI
        return d

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

    def parse_csv_file(self):
        self.file_path_train=[]
        self.file_path_val=[]
        self.file_path_test=[]
        random.seed(3407)
        train_big_block=range(0,800)
        val_big_block=range(800,900)
        test_big_block=range(900,1000)
        train_small_piece=random.sample(range(0,17*17),50)
        remain_small_piece=list(set(list(range(0,17*17)))-set(train_small_piece))
        val_small_piece=random.sample(remain_small_piece,50)
        test_small_piece=random.sample(list(set(remain_small_piece)-set(val_small_piece)),50)
        i_6_list=random.sample(range(0,20),4)
        for i in train_big_block:
            i_1=i//100
            i_2=(i%100)//10
            i_3=(i%100)%10
            for t in train_small_piece:
                i_4=t//17
                i_5=t%17
                if 17>=i_4>=2 and 17>=i_5>=2:
                    for i_6 in i_6_list:
                        self.file_path_train.append([i_1,i_2,i_3,i_4,i_5,i_6])
        for i in val_big_block:
            i_1=i//100
            i_2=(i%100)//10
            i_3=(i%100)%10
            for t in val_small_piece:
                i_4=t//17
                i_5=t%17
                if 17>=i_4>=2 and 17>=i_5>=2:
                    for i_6 in i_6_list:
                        self.file_path_val.append([i_1,i_2,i_3,i_4,i_5,i_6])
        for i in test_big_block:
            i_1=i//100
            i_2=(i%100)//10
            i_3=(i%100)%10
            for t in test_small_piece:
                i_4=t//17
                i_5=t%17
                if 17>=i_4>=2 and 17>=i_5>=2:
                    for i_6 in i_6_list:
                        self.file_path_test.append([i_1,i_2,i_3,i_4,i_5,i_6])
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

    def _load_one_image(self, d,image_path):
        final_3d=d[0,image_path[0]*20:image_path[0]*20+20,image_path[1]*20:image_path[1]*20+20,image_path[2]*20:image_path[2]*20+20]
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
            images_3d[i, :, :, :, 0] = self._load_one_image(self.imaged,self.file_path_train[idxlist[i]])
            images_2d_list.append([self.file_path_train[idxlist[i]][3],self.file_path_train[idxlist[i]][4],self.file_path_train[idxlist[i]][5]])
        return images_3d,images_2d_list

    def _load_batch_image_test(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_test))
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list=[]
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.imaged,self.file_path_test[idxlist[i]])
            images_2d_list.append([self.file_path_test[idxlist[i]][3],self.file_path_test[idxlist[i]][4],self.file_path_test[idxlist[i]][5]])
        return images_3d,images_2d_list

    def _load_batch_image_val(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_val))
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list=[]
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.imaged,self.file_path_val[idxlist[i]])
            images_2d_list.append([self.file_path_val[idxlist[i]][3],self.file_path_val[idxlist[i]][4],self.file_path_val[idxlist[i]][5]])
        return images_3d,images_2d_list

