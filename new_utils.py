import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data1(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    #print("{} images were found in the dataset.".format(sum(every_class_num)))
    #print("{} images for training.".format(len(train_images_path)))
    #print("{} images for validation.".format(len(val_images_path)))
    #assert len(train_images_path) > 0, "number of training images must greater than 0."
    #assert len(val_images_path) > 0, "number of validation images must greater than 0."

    '''plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()'''

    return test_images_path, test_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model1,model2, optimizer, data_loader, device, epoch):
    model1.train()
    model2.train()
    loss_function1 = torch.nn.CrossEntropyLoss()
    loss_function2 = torch.nn.CrossEntropyLoss()
    #loss_function3 = torch.nn.MSELoss()
    loss_function3 = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = 0   # 累计预测正确的样本数
    optimizer.zero_grad()
    acc1=0
    acc2=0
    acc3=0
    sample_num = 0
    accumulation_steps=16
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images3d, images2d, labels1,labels2,labels3 = data
        #print("3d",images3d.shape)
        #print("2d",images2d.shape)
        #print(labels.shape)
        #print(images.shape)
        images2d=torch.unsqueeze(images2d,dim=1).float()
        #labels=torch.squeeze(labels,dim=0)
        images3d=images3d.float()
        #print(images3d.shape)
        #print(images2d.shape)
        images3d = images3d.to(device)
        images2d = images2d.to(device)
        #print(images2d.shape)
        labels1=labels1.to(device)
        #print(labels1.shape)
        labels2=labels2.to(device)
        labels3=labels3.to(device)
        sample_num += images3d.shape[0]
        #print(images3d.shape)
        #images3d=torch.squeeze(images3d, dim=5)
        x,k,v = model1(images2d.to(device))
        pred_list,pred_class=model2(images3d.to(device),k,v)
        #print("pred",pred_list)
        #print("labels1",labels1)
        #print("labels2",labels2)
        #print("labels3",labels3)
        pred_class_1=pred_class[0]
        pred_class_2=pred_class[1]
        pred_class_3=pred_class[2]
        pred_class_4=pred_class[3]
        pred_class_5=pred_class[4]
        pred_class_6=pred_class[5]
        pred_class_7=pred_class[6]
        pred_class_8=pred_class[7]
        pred_class_9=pred_class[8]
        pred_class_10=pred_class[9]
        pred_class_11=pred_class[10]
        pred_class_12=pred_class[11]
        pred_class_13=pred_class[12]
        pred_class_14=pred_class[13]
        pred_class_15=pred_class[14]
        pred_class_16=pred_class[15]
        pred_class_17=pred_class[16]
        pred_class_18=pred_class[17]
        pred1_1=pred_list[0]
        pred1_2=pred_list[1]
        pred1_3=pred_list[2]
        pred2_1=pred_list[3]
        pred2_2=pred_list[4]
        pred2_3=pred_list[5]
        pred3_1=pred_list[6]
        pred3_2=pred_list[7]
        pred3_3=pred_list[8]
        pred3_4=pred_list[9]
        pred3_5=pred_list[10]
        pred3_6=pred_list[11]
        pred3_7=pred_list[12]
        pred3_8=pred_list[13]
        pred3_9=pred_list[14]
        pred3_10=pred_list[15]
        pred3_11=pred_list[16]
        pred3_12=pred_list[17]
        #print(labels3)
        true1_1=torch.unsqueeze(labels1[0][0],dim=0)
        true1_2=torch.unsqueeze(labels1[0][1],dim=0)
        true1_3=torch.unsqueeze(labels1[0][1],dim=0)
        true2_1=torch.unsqueeze(labels2[0][0],dim=0)
        true2_2=torch.unsqueeze(labels2[0][1],dim=0)
        true2_3=torch.unsqueeze(labels2[0][2],dim=0)
        true3_1=torch.unsqueeze(labels3[0][0],dim=0)
        true3_2=torch.unsqueeze(labels3[0][1],dim=0)
        true3_3=torch.unsqueeze(labels3[0][2],dim=0)
        true3_4=torch.unsqueeze(labels3[0][3],dim=0)
        true3_5=torch.unsqueeze(labels3[0][4],dim=0)
        true3_6=torch.unsqueeze(labels3[0][5],dim=0)
        true3_7=torch.unsqueeze(labels3[0][6],dim=0)
        true3_8=torch.unsqueeze(labels3[0][7],dim=0)
        true3_9=torch.unsqueeze(labels3[0][8],dim=0)
        true3_10=torch.unsqueeze(labels3[0][9],dim=0)
        true3_11=torch.unsqueeze(labels3[0][10],dim=0)
        true3_12=torch.unsqueeze(labels3[0][11],dim=0)
        #print(pred2_1,true2_1)
        #print(true1_1.view(-1,1))
        #print("equal",(torch.eq(pred1_1,true1_1.view(-1,1)).sum()))
        #print("pred",pred3_1)
        #print(true3_1.view(-1,1))
        #print("equal",(torch.eq(pred3_1,true3_1.view(-1,1)).sum()+torch.eq(pred3_2,true3_2.view(-1,1)).sum()+torch.eq(pred3_3,true3_3.view(-1,1)).sum())==3)
        #print(labels1)
        #print("before",labels)
        #print("after",pred_classes)
        if (torch.eq(pred1_1,true1_1).sum()+torch.eq(pred1_2,true1_2).sum()+torch.eq(pred1_3,true1_3).sum())==3:
            accu_num += 0.5
            acc1+=1
            if (torch.eq(pred2_1,true2_1.view(-1,1)).sum()+torch.eq(pred2_2,true2_2.view(-1,1)).sum()+torch.eq(pred2_3,true2_3.view(-1,1)).sum())==3:
                accu_num+=0.3
                acc2+=1
                if (((torch.eq(pred3_1,true3_1.view(-1,1)).sum()+torch.eq(pred3_2,true3_2.view(-1,1)).sum()+torch.eq(pred3_3,true3_3.view(-1,1)).sum())==3) and ((torch.eq(pred3_1,true3_4.view(-1,1)).sum()+torch.eq(pred3_2,true3_5.view(-1,1)).sum()+torch.eq(pred3_3,true3_6.view(-1,1)).sum())==3) and ((torch.eq(pred3_1,true3_7.view(-1,1)).sum()+torch.eq(pred3_2,true3_8.view(-1,1)).sum()+torch.eq(pred3_3,true3_9.view(-1,1)).sum()==3)) and ((torch.eq(pred3_1,true3_10.view(-1,1)).sum()+torch.eq(pred3_2,true3_11.view(-1,1)).sum()+torch.eq(pred3_3,true3_12.view(-1,1)).sum())==3)):
                    accu_num += 0.2
                    acc3+=1
        #true1_1=true1_1.long()
        #true1_2=true1_2.long()
        #true1_3=true1_3.long()
        #print(true1_1)
        #print(true1_1.dtype)
        #print(pred1_1.dtype)
        #print(true1_1.shape)
        #print(pred_class_1.shape)
        #print(pred_class_1)
        #print(true1_1.dtype)
        loss1 = (loss_function1(pred_class_1,true1_1)+loss_function1(pred_class_2,true1_2)+loss_function1(pred_class_3,true1_3))/3
        loss2 = (loss_function2(pred_class_4, true2_1)+loss_function2(pred_class_5, true2_2)+loss_function2(pred_class_6, true2_3))/3
        '''loss3 = (loss_function3(pred3_1, true3_1)+loss_function3(pred3_2, true3_2)+loss_function3(pred3_3, true3_3)
        +loss_function3(pred3_4, true3_4)+loss_function3(pred3_5, true3_5)+loss_function3(pred3_6, true3_6)
        +loss_function3(pred3_7, true3_7)+loss_function3(pred3_8, true3_8)+loss_function3(pred3_9, true3_9)
        +loss_function3(pred3_10, true3_10)+loss_function3(pred3_11, true3_11)+loss_function3(pred3_12, true3_12))/12'''
        loss3 = (loss_function3(pred_class_7, true3_1)+loss_function3(pred_class_8, true3_2)+loss_function3(pred_class_9, true3_3)
        +loss_function3(pred_class_10, true3_4)+loss_function3(pred_class_11, true3_5)+loss_function3(pred_class_12, true3_6)
        +loss_function3(pred_class_13, true3_7)+loss_function3(pred_class_14, true3_8)+loss_function3(pred_class_15, true3_9)
        +loss_function3(pred_class_16, true3_10)+loss_function3(pred_class_17, true3_11)+loss_function3(pred_class_18, true3_12))/12
        loss=0.2*loss1+0.3*loss2+0.5*loss3
        #print(loss.dtype)
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=10, norm_type=2)
        accu_loss += loss.detach()
        data_loader.desc = "[train {}] l:{:.3f}, acc:{:.3f}, acc1:{:.3f}, acc2:{:.5f}({:.0f}), acc3:{:.5f}({:.0f})".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num / sample_num,
                                                                               acc1/sample_num,
                                                                               acc2/sample_num,
                                                                               acc2,
                                                                               acc3/sample_num,acc3)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        if ((step+1)%accumulation_steps)==0:
            optimizer.step()
            optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num / sample_num,acc1/sample_num,acc2/sample_num,acc3/sample_num
class LabelSmoothingCrossEntropy(torch.nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
'''def train_one_epoch(model, optimizer, data, device, epoch, round):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    #loss_function=LabelSmoothingCrossEntropy()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    #desc="[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(0,0,0)
    round = tqdm(range(round), file=sys.stdout)
    for step,d in enumerate(round):
        images, labels = data[step]
        images = images.to(device)
        labels=labels.to(device)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        #print(pred_classes)
        #print(pred_classes.size)
        #print(labels)
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        #print("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,accu_loss.item() / (step + 1),accu_num.item() / sample_num))
        round.desc="[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num'''

@torch.no_grad()
def evaluate(model1,model2, data_loader, device, epoch):
    model1.eval()
    model2.eval()
    loss_function1 = torch.nn.CrossEntropyLoss()
    loss_function2 = torch.nn.CrossEntropyLoss()
    #loss_function3 = torch.nn.MSELoss()
    loss_function3 = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = 0   # 累计预测正确的样本
    acc1=0
    acc2=0
    acc3=0
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images3d, images2d, labels1,labels2,labels3 = data
        #print(labels.shape)
        #print(images.shape)
        images2d=torch.unsqueeze(images2d,dim=1).float()
        #labels=torch.squeeze(labels,dim=0)
        images3d=images3d.float()
        images3d = images3d.to(device)
        images2d = images2d.to(device)
        #print(images2d.shape)
        labels1=labels1.to(device)
        labels2=labels2.to(device)
        labels3=labels3.to(device)
        sample_num += images3d.shape[0]
        #print(images3d.shape)
        #images3d=torch.squeeze(images3d, dim=5)
        x,k,v = model1(images2d.to(device))
        pred_list,pred_class=model2(images3d.to(device),k,v)
        pred_class_1=pred_class[0]
        pred_class_2=pred_class[1]
        pred_class_3=pred_class[2]
        pred_class_4=pred_class[3]
        pred_class_5=pred_class[4]
        pred_class_6=pred_class[5]
        pred_class_7=pred_class[6]
        pred_class_8=pred_class[7]
        pred_class_9=pred_class[8]
        pred_class_10=pred_class[9]
        pred_class_11=pred_class[10]
        pred_class_12=pred_class[11]
        pred_class_13=pred_class[12]
        pred_class_14=pred_class[13]
        pred_class_15=pred_class[14]
        pred_class_16=pred_class[15]
        pred_class_17=pred_class[16]
        pred_class_18=pred_class[17]
        pred1_1=pred_list[0]
        pred1_2=pred_list[1]
        pred1_3=pred_list[2]
        pred2_1=pred_list[3]
        pred2_2=pred_list[4]
        pred2_3=pred_list[5]
        pred3_1=pred_list[6]
        pred3_2=pred_list[7]
        pred3_3=pred_list[8]
        pred3_4=pred_list[9]
        pred3_5=pred_list[10]
        pred3_6=pred_list[11]
        pred3_7=pred_list[12]
        pred3_8=pred_list[13]
        pred3_9=pred_list[14]
        pred3_10=pred_list[15]
        pred3_11=pred_list[16]
        pred3_12=pred_list[17]
        #print(labels3)
        true1_1=torch.unsqueeze(labels1[0][0],dim=0)
        true1_2=torch.unsqueeze(labels1[0][1],dim=0)
        true1_3=torch.unsqueeze(labels1[0][1],dim=0)
        true2_1=torch.unsqueeze(labels2[0][0],dim=0)
        true2_2=torch.unsqueeze(labels2[0][1],dim=0)
        true2_3=torch.unsqueeze(labels2[0][2],dim=0)
        true3_1=torch.unsqueeze(labels3[0][0],dim=0)
        true3_2=torch.unsqueeze(labels3[0][1],dim=0)
        true3_3=torch.unsqueeze(labels3[0][2],dim=0)
        true3_4=torch.unsqueeze(labels3[0][3],dim=0)
        true3_5=torch.unsqueeze(labels3[0][4],dim=0)
        true3_6=torch.unsqueeze(labels3[0][5],dim=0)
        true3_7=torch.unsqueeze(labels3[0][6],dim=0)
        true3_8=torch.unsqueeze(labels3[0][7],dim=0)
        true3_9=torch.unsqueeze(labels3[0][8],dim=0)
        true3_10=torch.unsqueeze(labels3[0][9],dim=0)
        true3_11=torch.unsqueeze(labels3[0][10],dim=0)
        true3_12=torch.unsqueeze(labels3[0][11],dim=0)
        #print(pred1_1)
        #print(labels1)
        #print("before",labels)
        #print("after",pred_classes)
        if (torch.eq(pred1_1,true1_1).sum()+torch.eq(pred1_2,true1_2).sum()+torch.eq(pred1_3,true1_3).sum())==3:
            accu_num += 0.5
            acc1+=1
            if (torch.eq(pred2_1,true2_1.view(-1,1)).sum()+torch.eq(pred2_2,true2_2.view(-1,1)).sum()+torch.eq(pred2_3,true2_3.view(-1,1)).sum())==3:
                accu_num+=0.3
                acc2+=1
                if (((torch.eq(pred3_1,true3_1.view(-1,1)).sum()+torch.eq(pred3_2,true3_2.view(-1,1)).sum()+torch.eq(pred3_3,true3_3.view(-1,1)).sum())==3) and ((torch.eq(pred3_1,true3_4.view(-1,1)).sum()+torch.eq(pred3_2,true3_5.view(-1,1)).sum()+torch.eq(pred3_3,true3_6.view(-1,1)).sum())==3) and ((torch.eq(pred3_1,true3_7.view(-1,1)).sum()+torch.eq(pred3_2,true3_8.view(-1,1)).sum()+torch.eq(pred3_3,true3_9.view(-1,1)).sum()==3)) and ((torch.eq(pred3_1,true3_10.view(-1,1)).sum()+torch.eq(pred3_2,true3_11.view(-1,1)).sum()+torch.eq(pred3_3,true3_12.view(-1,1)).sum())==3)):
                    accu_num += 0.2
                    acc3+=1
        #true1_1=true1_1.long()
        #true1_2=true1_2.long()
        #true1_3=true1_3.long()
        #print(true1_1)
        #print(true1_1.dtype)
        #print(pred1_1.dtype)
        #print(true1_1.shape)
        #print(pred_class_1.shape)
        #print(pred_class_1)
        #print(true1_1)
        loss1 = (loss_function1(pred_class_1,true1_1)+loss_function1(pred_class_2,true1_2)+loss_function1(pred_class_3,true1_3))/3
        loss2 = (loss_function2(pred_class_4, true2_1)+loss_function2(pred_class_5, true2_2)+loss_function2(pred_class_6, true2_3))/3
        '''loss3 = (loss_function3(pred3_1, true3_1)+loss_function3(pred3_2, true3_2)+loss_function3(pred3_3, true3_3)
        +loss_function3(pred3_4, true3_4)+loss_function3(pred3_5, true3_5)+loss_function3(pred3_6, true3_6)
        +loss_function3(pred3_7, true3_7)+loss_function3(pred3_8, true3_8)+loss_function3(pred3_9, true3_9)
        +loss_function3(pred3_10, true3_10)+loss_function3(pred3_11, true3_11)+loss_function3(pred3_12, true3_12))/12'''
        loss3 = (loss_function3(pred_class_7, true3_1)+loss_function3(pred_class_8, true3_2)+loss_function3(pred_class_9, true3_3)
        +loss_function3(pred_class_10, true3_4)+loss_function3(pred_class_11, true3_5)+loss_function3(pred_class_12, true3_6)
        +loss_function3(pred_class_13, true3_7)+loss_function3(pred_class_14, true3_8)+loss_function3(pred_class_15, true3_9)
        +loss_function3(pred_class_16, true3_10)+loss_function3(pred_class_17, true3_11)+loss_function3(pred_class_18, true3_12))/12
        loss=0.2*loss1+0.3*loss2+0.5*loss3
        accu_loss += loss.detach()
        data_loader.desc = "[val {}] l:{:.3f}, acc:{:.3f}, acc1:{:.3f}, acc2:{:.5f}({:.0f}), acc3:{:.5f}({:.0f})".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num / sample_num,
                                                                               acc1/sample_num,
                                                                               acc2/sample_num,
                                                                               acc2,
                                                                               acc3/sample_num,acc3)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1),accu_num / sample_num,acc1/sample_num,acc2/sample_num,acc3/sample_num
'''@torch.no_grad()
def evaluate(model, data, device, epoch, round):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    round = tqdm(range(round), file=sys.stdout)
    #data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step,d in enumerate(round):
            #print("This is round",step)
            images, labels = data[step]
            images = images.to(device)
            labels=labels.to(device)
            sample_num += images.shape[0]

            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            accu_loss += loss
            round.desc="[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num'''
@torch.no_grad()
def test(model1,model2, data_loader, device, epoch):
    model1.eval()
    model2.eval()
    loss_function1 = torch.nn.CrossEntropyLoss()
    loss_function2 = torch.nn.CrossEntropyLoss()
    #loss_function3 = torch.nn.MSELoss()
    loss_function3 = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = 0   # 累计预测正确的样本数
    acc1=0
    acc2=0
    acc3=0
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images3d, images2d, labels1,labels2,labels3 = data
        #print(labels.shape)
        #print(images.shape)
        images2d=torch.unsqueeze(images2d,dim=1).float()
        #labels=torch.squeeze(labels,dim=0)
        images3d=images3d.float()
        images3d = images3d.to(device)
        images2d = images2d.to(device)
        #print(images2d.shape)
        labels1=labels1.to(device)
        labels2=labels2.to(device)
        labels3=labels3.to(device)
        sample_num += images3d.shape[0]
        #print(images3d.shape)
        #images3d=torch.squeeze(images3d, dim=5)
        x,k,v = model1(images2d.to(device))
        pred_list,pred_class=model2(images3d.to(device),k,v)
        pred_class_1=pred_class[0]
        pred_class_2=pred_class[1]
        pred_class_3=pred_class[2]
        pred_class_4=pred_class[3]
        pred_class_5=pred_class[4]
        pred_class_6=pred_class[5]
        pred_class_7=pred_class[6]
        pred_class_8=pred_class[7]
        pred_class_9=pred_class[8]
        pred_class_10=pred_class[9]
        pred_class_11=pred_class[10]
        pred_class_12=pred_class[11]
        pred_class_13=pred_class[12]
        pred_class_14=pred_class[13]
        pred_class_15=pred_class[14]
        pred_class_16=pred_class[15]
        pred_class_17=pred_class[16]
        pred_class_18=pred_class[17]
        pred1_1=pred_list[0]
        pred1_2=pred_list[1]
        pred1_3=pred_list[2]
        pred2_1=pred_list[3]
        pred2_2=pred_list[4]
        pred2_3=pred_list[5]
        pred3_1=pred_list[6]
        pred3_2=pred_list[7]
        pred3_3=pred_list[8]
        pred3_4=pred_list[9]
        pred3_5=pred_list[10]
        pred3_6=pred_list[11]
        pred3_7=pred_list[12]
        pred3_8=pred_list[13]
        pred3_9=pred_list[14]
        pred3_10=pred_list[15]
        pred3_11=pred_list[16]
        pred3_12=pred_list[17]
        #print(labels3)
        true1_1=torch.unsqueeze(labels1[0][0],dim=0)
        true1_2=torch.unsqueeze(labels1[0][1],dim=0)
        true1_3=torch.unsqueeze(labels1[0][1],dim=0)
        true2_1=torch.unsqueeze(labels2[0][0],dim=0)
        true2_2=torch.unsqueeze(labels2[0][1],dim=0)
        true2_3=torch.unsqueeze(labels2[0][2],dim=0)
        true3_1=torch.unsqueeze(labels3[0][0],dim=0)
        true3_2=torch.unsqueeze(labels3[0][1],dim=0)
        true3_3=torch.unsqueeze(labels3[0][2],dim=0)
        true3_4=torch.unsqueeze(labels3[0][3],dim=0)
        true3_5=torch.unsqueeze(labels3[0][4],dim=0)
        true3_6=torch.unsqueeze(labels3[0][5],dim=0)
        true3_7=torch.unsqueeze(labels3[0][6],dim=0)
        true3_8=torch.unsqueeze(labels3[0][7],dim=0)
        true3_9=torch.unsqueeze(labels3[0][8],dim=0)
        true3_10=torch.unsqueeze(labels3[0][9],dim=0)
        true3_11=torch.unsqueeze(labels3[0][10],dim=0)
        true3_12=torch.unsqueeze(labels3[0][11],dim=0)
        #print(pred1_1)
        #print(labels1)
        #print("before",labels)
        #print("after",pred_classes)
        if (torch.eq(pred1_1,true1_1).sum()+torch.eq(pred1_2,true1_2).sum()+torch.eq(pred1_3,true1_3).sum())==3:
            accu_num += 0.5
            acc1+=1
            if (torch.eq(pred2_1,true2_1.view(-1,1)).sum()+torch.eq(pred2_2,true2_2.view(-1,1)).sum()+torch.eq(pred2_3,true2_3.view(-1,1)).sum())==3:
                accu_num+=0.3
                acc2+=1
                if (((torch.eq(pred3_1,true3_1.view(-1,1)).sum()+torch.eq(pred3_2,true3_2.view(-1,1)).sum()+torch.eq(pred3_3,true3_3.view(-1,1)).sum())==3) and ((torch.eq(pred3_1,true3_4.view(-1,1)).sum()+torch.eq(pred3_2,true3_5.view(-1,1)).sum()+torch.eq(pred3_3,true3_6.view(-1,1)).sum())==3) and ((torch.eq(pred3_1,true3_7.view(-1,1)).sum()+torch.eq(pred3_2,true3_8.view(-1,1)).sum()+torch.eq(pred3_3,true3_9.view(-1,1)).sum()==3)) and ((torch.eq(pred3_1,true3_10.view(-1,1)).sum()+torch.eq(pred3_2,true3_11.view(-1,1)).sum()+torch.eq(pred3_3,true3_12.view(-1,1)).sum())==3)):
                    accu_num += 0.2
                    acc3+=1
        #true1_1=true1_1.long()
        #true1_2=true1_2.long()
        #true1_3=true1_3.long()
        #print(true1_1)
        #print(true1_1.dtype)
        #print(pred1_1.dtype)
        #print(true1_1.shape)
        #print(pred_class_1.shape)
        #print(pred_class_1)
        #print(true1_1)
        loss1 = (loss_function1(pred_class_1,true1_1)+loss_function1(pred_class_2,true1_2)+loss_function1(pred_class_3,true1_3))/3
        loss2 = (loss_function2(pred_class_4, true2_1)+loss_function2(pred_class_5, true2_2)+loss_function2(pred_class_6, true2_3))/3
        '''loss3 = (loss_function3(pred3_1, true3_1)+loss_function3(pred3_2, true3_2)+loss_function3(pred3_3, true3_3)
        +loss_function3(pred3_4, true3_4)+loss_function3(pred3_5, true3_5)+loss_function3(pred3_6, true3_6)
        +loss_function3(pred3_7, true3_7)+loss_function3(pred3_8, true3_8)+loss_function3(pred3_9, true3_9)
        +loss_function3(pred3_10, true3_10)+loss_function3(pred3_11, true3_11)+loss_function3(pred3_12, true3_12))/12'''
        loss3 = (loss_function3(pred_class_7, true3_1)+loss_function3(pred_class_8, true3_2)+loss_function3(pred_class_9, true3_3)
        +loss_function3(pred_class_10, true3_4)+loss_function3(pred_class_11, true3_5)+loss_function3(pred_class_12, true3_6)
        +loss_function3(pred_class_13, true3_7)+loss_function3(pred_class_14, true3_8)+loss_function3(pred_class_15, true3_9)
        +loss_function3(pred_class_16, true3_10)+loss_function3(pred_class_17, true3_11)+loss_function3(pred_class_18, true3_12))/12
        loss=0.2*loss1+0.3*loss2+0.5*loss3
        accu_loss += loss.detach()
        data_loader.desc = "[test {}] l:{:.3f}, acc:{:.3f}, acc1:{:.3f}, acc2:{:.5f}({:.0f}), acc3:{:.5f}({:.0f})".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num / sample_num,
                                                                               acc1/sample_num,
                                                                               acc2/sample_num,
                                                                               acc2,
                                                                               acc3/sample_num,acc3)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num / sample_num,acc1/sample_num,acc2/sample_num,acc3/sample_num

'''def test(model, data, device, epoch, round):
    #loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    #accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    round = tqdm(range(round), file=sys.stdout)
    #data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step,d in enumerate(round):
            images, labels = data[step]
            images = images.to(device)
            labels=labels.to(device)
            sample_num += images.shape[0]

            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            #loss = loss_function(pred, labels.to(device))
            #accu_loss += loss
            round.desc="[test epoch {}] acc: {:.3f}".format(epoch,accu_num.item() / sample_num)
            #data_loader.desc = "[test epoch {}] acc: {:.3f}".format(epoch,accu_num.item() / sample_num)
    return accu_num.item() / sample_num'''

def test_model(model, data, device, epoch, round):
    #loss_function = torch.nn.CrossEntropyLoss()

    #model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    #accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    round = tqdm(range(round), file=sys.stdout)
    #data_loader = tqdm(data_loader, file=sys.stdout)
    for step,d in enumerate(round):
        images, labels = data[step]
        images = images.to(device)
        labels=labels.to(device)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        #loss = loss_function(pred, labels.to(device))
        #accu_loss += loss
        round.desc="[test epoch {}] acc: {:.3f}".format(epoch,accu_num.item() / sample_num)
        #data_loader.desc = "[test epoch {}] acc: {:.3f}".format(epoch,accu_num.item() / sample_num)
    return accu_num.item() / sample_num
    
