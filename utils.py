#This is for AsymmetricLoss
import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
from special_loss import AsymmetricLossOptimized
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model1,model2,model3, optimizer,optimizer2, data_loader, device, epoch):
    model1.train()
    model2.train()
    model3.train()
    loss_function1=AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    accu_loss = torch.zeros(1).to(device)  
    accu_loss2= torch.zeros(1).to(device)
    ac=0
    accu_num = 0   
    acc_num1=0
    optimizer.zero_grad()
    optimizer2.zero_grad()
    sample_num = 0
    accumulation_steps=1
    data_loader = tqdm(data_loader, file=sys.stdout)
    scaler = GradScaler()
    for step, data in enumerate(data_loader):
        images3d, images2d, labels1,labels1_loss1,labels2,labels2_loss = data
        images3d=images3d.float()
        images2d=images2d.float()
        images3d=torch.squeeze(images3d,dim=5)
        images3d=torch.squeeze(images3d,dim=0)
        images2d=torch.squeeze(images2d,dim=0)
        images3d=torch.unsqueeze(images3d,dim=1)
        images2d=torch.unsqueeze(images2d,dim=1)
        images3d = images3d.to(device)
        images2d = images2d.to(device)
        labels1_loss1=torch.squeeze(labels1_loss1,(0,2))
        sample_num += images3d.shape[0]
        with autocast():
            x,k,v = model1(images2d.to(device))
            pred1,pred_class,pred_class_k1,x_set=model2(images3d.to(device),k,v)
        loss1_1 = loss_function1(pred1, labels1_loss1.to(device))
        loss1_1=scaler.scale(loss1_1)
        loss=loss1_1
        loss=scaler.scale(loss)
        loss.backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        next_block_list=[]
        position_list=[]
        pred_class_list=[]
        for i in range(len(labels1)):
            temp_pred_list=[]
            for t in range(len(pred_class[i])):
                temp_pred_list.append(pred_class[i][t].item())
            pred_class_list.append(temp_pred_list)
        for i in range(len(labels1)):
            labels1_list=[]
            for j in range(len(labels1[i])):
                labels1_list.append(labels1[i][j].item())
            find_correct_block=False
            for t in range(len(pred_class_list[i])):
                if pred_class_list[i][t] in labels1_list:
                    next_block=pred_class_list[i][t]
                    find_correct_block=True
                    position_list.append(labels1_list.index(pred_class_list[i][t]))
                    break
            if find_correct_block==False:
                next_block=pred_class_list[i][0]
                position_list.append(-1)
            next_block_list.append(next_block)
        images3d_new=torch.zeros((len(labels1), 1,10,10,10))         #Low erro predict block size, we use 10*10*10 here
        labels2_true=[]
        labels2_loss_true1=torch.zeros((len(labels1),27))           #Label number, here we have 27 labels
        for i in range(len(next_block_list)):
            if position_list[i]!=-1:
                labels2_true.append(labels2[i][position_list[i]])
                for t in range(len(labels2_loss[i][position_list[i]][0])):
                    if labels2_loss[i][position_list[i]][0][t].item()==1:
                        labels2_loss_true1[i][t]=1
            else:
                a=torch.zeros((1))
                a[0]=-1 
                labels2_true.append([a])
            a=next_block_list[i]//9
            b=(next_block_list[i]-9*a)//3
            c=next_block_list[i]-9*a-3*b
            images3d_new[i]=images3d[i,:,a*5:a*5+10,b*5:b*5+10,c*5:c*5+10]
        images3d_new = images3d_new.to(device)
        with autocast():
            pred_new1,pred_class_new,pred_class_k1_new=model3(images3d_new.to(device),k,v)
        pred_class_list_new=[]
        for i in range(len(labels1)):
            temp_pred_list_new=[]
            for t in range(len(pred_class_new[i])):
                temp_pred_list_new.append(pred_class_new[i][t].item())
            pred_class_list_new.append(temp_pred_list_new)
        for i in range(len(labels1)):
            find=False
            f_ac=False
            for t in range(len(labels1[i])):
                if labels1[i][t].item() in pred_class_list[i]:
                    if f_ac==False:
                        ac+=1
                        f_ac=True
                    for k in range(len(labels2_true[i])):
                        if labels2_true[i][k].item() in pred_class_list_new[i]:
                            accu_num+=1
                            find=True
                            break
                if find==True:
                    break
        for i in range(len(labels1)):
            find=False
            for t in range(len(labels1[i])):
                if labels1[i][t].item() == pred_class_k1[i][0].item():
                    for k in range(len(labels2_true[i])):
                        if labels2_true[i][k].item() == pred_class_k1_new[i].item():
                            acc_num1+=1
                            find=True
                            break
                if find==True:
                    break
        loss2_1=loss_function1(pred_new1, labels2_loss_true1.to(device))
        loss2_1=scaler.scale(loss2_1)
        loss2=loss2_1
        loss2=scaler.scale(loss2)
        loss2.backward(retain_graph=True)
        if ((step+1)%accumulation_steps)==0:
            scaler.step(optimizer2)
            scaler.update()
            optimizer2.zero_grad()
        torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=model3.parameters(), max_norm=10, norm_type=2)
        accu_loss += loss.detach()
        accu_loss2+=loss2.detach()
        data_loader.desc = "[train {}] l:{:.3f},l2:{:.3f}, acc1:{:.3f},acc:{:.3f},acck1:{:.3f}".format(epoch,
                                                                    accu_loss.item()/(step + 1),
                                                                    accu_loss2.item() / (step + 1),
                                                                    ac/sample_num,
                                                                    accu_num / sample_num,
                                                                    acc_num1/sample_num
                                                                    )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1),accu_loss2.item()/(step + 1),ac/sample_num, accu_num / sample_num,acc_num1/sample_num,x_set

@torch.no_grad()
def evaluate(model1,model2,model3, data_loader, device, epoch):
    model1.eval()
    model2.eval()
    model3.eval()
    loss_function1=AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    accu_loss = torch.zeros(1).to(device)  
    accu_loss2= torch.zeros(1).to(device)
    accu_num = 0 
    acc_num1=0
    ac=0
    sample_num = 0
    scaler = GradScaler()
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images3d, images2d, labels1,labels1_loss1,labels2,labels2_loss = data
            images3d=images3d.float()
            images2d=images2d.float()
            images3d=torch.squeeze(images3d,dim=5)
            images3d=torch.squeeze(images3d,dim=0)
            images2d=torch.squeeze(images2d,dim=0)
            images3d=torch.unsqueeze(images3d,dim=1)
            images2d=torch.unsqueeze(images2d,dim=1)
            images3d = images3d.to(device)
            images2d = images2d.to(device)
            labels1_loss1=torch.squeeze(labels1_loss1,(0,2))
            sample_num += images3d.shape[0]
            with autocast():
                x,k,v = model1(images2d.to(device))
                pred1,pred_class,pred_class_k1,x_set=model2(images3d.to(device),k,v)
            loss1_1 = loss_function1(pred1, labels1_loss1.to(device))
            loss1_1=scaler.scale(loss1_1)
            loss=loss1_1
            loss=scaler.scale(loss)
            next_block_list=[]
            position_list=[]
            pred_class_list=[]
            for i in range(len(labels1)):
                temp_pred_list=[]
                for t in range(len(pred_class[i])):
                    temp_pred_list.append(pred_class[i][t].item())
                pred_class_list.append(temp_pred_list)
            for i in range(len(labels1)):
                labels1_list=[]
                for j in range(len(labels1[i])):
                    labels1_list.append(labels1[i][j].item())
                find_correct_block=False
                for t in range(len(pred_class_list[i])):
                    if pred_class_list[i][t] in labels1_list:
                        next_block=pred_class_list[i][t]
                        find_correct_block=True
                        position_list.append(labels1_list.index(pred_class_list[i][t]))
                        break
                if find_correct_block==False:
                    next_block=pred_class_list[i][0]
                    position_list.append(-1)
                next_block_list.append(next_block)
            images3d_new=torch.zeros((len(labels1), 1,10,10,10))
            labels2_true=[]
            labels2_loss_true1=torch.zeros((len(labels1),27))
            for i in range(len(next_block_list)):
                if position_list[i]!=-1:
                    labels2_true.append(labels2[i][position_list[i]])
                    for t in range(len(labels2_loss[i][position_list[i]])):
                        if labels2_loss[i][position_list[i]][0][t].item()==1:
                            labels2_loss_true1[i][t]=1
                else:
                    a=torch.zeros((1))
                    a[0]=-1 
                    labels2_true.append([a])
                a=next_block_list[i]//9
                b=(next_block_list[i]-9*a)//3
                c=next_block_list[i]-9*a-3*b
                images3d_new[i]=images3d[i,:,a*5:a*5+10,b*5:b*5+10,c*5:c*5+10]
            images3d_new = images3d_new.to(device)
            with autocast():
                pred_new1,pred_class_new,pred_class_k1_new=model3(images3d_new.to(device),k,v)
            pred_class_list_new=[]
            for i in range(len(labels1)):
                temp_pred_list_new=[]
                for t in range(len(pred_class_new[i])):
                    temp_pred_list_new.append(pred_class_new[i][t].item())
                pred_class_list_new.append(temp_pred_list_new)
            for i in range(len(labels1)):
                find=False
                f_ac=False
                for t in range(len(labels1[i])):
                    if labels1[i][t].item() in pred_class_list[i]:
                        if f_ac==False:
                            ac+=1
                            f_ac=True
                        for k in range(len(labels2_true[i])):
                            if labels2_true[i][k].item() in pred_class_list_new[i]:
                                accu_num+=1
                                find=True
                                break
                    if find==True:
                        break
            for i in range(len(labels1)):
                find=False
                for t in range(len(labels1[i])):
                    if labels1[i][t].item() == pred_class_k1[i][0].item():
                        for k in range(len(labels2_true[i])):
                            if labels2_true[i][k].item() == pred_class_k1_new[i].item():
                                acc_num1+=1
                                find=True
                                break
                    if find==True:
                        break
            loss2_1=loss_function1(pred_new1, labels2_loss_true1.to(device))
            loss2_1=scaler.scale(loss2_1)
            loss2=loss2_1
            loss2=scaler.scale(loss2)
            torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(parameters=model3.parameters(), max_norm=10, norm_type=2)
            accu_loss += loss.detach()
            accu_loss2+=loss2.detach()
            data_loader.desc = "[val {}] l:{:.3f},l2:{:.3f}, acc1:{:.3f},acc:{:.3f},acck1:{:.3f}".format(epoch,
                                                                        accu_loss.item()/(step + 1),
                                                                        accu_loss2.item() / (step + 1),
                                                                        ac/sample_num,
                                                                        accu_num / sample_num,
                                                                        acc_num1/sample_num
                                                                        )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

    return accu_loss.item() / (step + 1),accu_loss2.item()/(step + 1),ac/sample_num, accu_num / sample_num,acc_num1/sample_num

@torch.no_grad()
def test(model1,model2,model3, data_loader, device, epoch):
    model1.eval()
    model2.eval()
    model3.eval()
    loss_function1=AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    accu_loss = torch.zeros(1).to(device) 
    accu_loss2= torch.zeros(1).to(device)
    accu_num = 0   
    acc_num1=0
    ac=0
    sample_num = 0
    scaler = GradScaler()
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images3d, images2d, labels1,labels1_loss1,labels2,labels2_loss = data
            images3d=images3d.float()
            images2d=images2d.float()
            images3d=torch.squeeze(images3d,dim=5)
            images3d=torch.squeeze(images3d,dim=0)
            images2d=torch.squeeze(images2d,dim=0)
            images3d=torch.unsqueeze(images3d,dim=1)
            images2d=torch.unsqueeze(images2d,dim=1)
            images3d = images3d.to(device)
            images2d = images2d.to(device)
            labels1_loss1=torch.squeeze(labels1_loss1,(0,2))
            sample_num += images3d.shape[0]
            with autocast():
                x,k,v = model1(images2d.to(device))
                pred1,pred_class,pred_class_k1,x_set=model2(images3d.to(device),k,v)
            loss1_1 = loss_function1(pred1, labels1_loss1.to(device))
            loss1_1=scaler.scale(loss1_1)
            loss=loss1_1
            loss=scaler.scale(loss)
            next_block_list=[]
            position_list=[]
            pred_class_list=[]
            for i in range(len(labels1)):
                temp_pred_list=[]
                for t in range(len(pred_class[i])):
                    temp_pred_list.append(pred_class[i][t].item())
                pred_class_list.append(temp_pred_list)
            for i in range(len(labels1)):
                labels1_list=[]
                for j in range(len(labels1[i])):
                    labels1_list.append(labels1[i][j].item())
                find_correct_block=False
                for t in range(len(pred_class_list[i])):
                    if pred_class_list[i][t] in labels1_list:
                        next_block=pred_class_list[i][t]
                        find_correct_block=True
                        position_list.append(labels1_list.index(pred_class_list[i][t]))
                        break
                if find_correct_block==False:
                    next_block=pred_class_list[i][0]
                    position_list.append(-1)
                next_block_list.append(next_block)
            images3d_new=torch.zeros((len(labels1), 1,10,10,10))
            labels2_true=[]
            labels2_loss_true1=torch.zeros((len(labels1),27))
            for i in range(len(next_block_list)):
                if position_list[i]!=-1:
                    labels2_true.append(labels2[i][position_list[i]])
                    for t in range(len(labels2_loss[i][position_list[i]])):
                        if labels2_loss[i][position_list[i]][0][t].item()==1:
                            labels2_loss_true1[i][t]=1
                else:
                    a=torch.zeros((1))
                    a[0]=-1 
                    labels2_true.append([a])
                a=next_block_list[i]//9
                b=(next_block_list[i]-9*a)//3
                c=next_block_list[i]-9*a-3*b
                images3d_new[i]=images3d[i,:,a*5:a*5+10,b*5:b*5+10,c*5:c*5+10]
            images3d_new = images3d_new.to(device)
            with autocast():
                pred_new1,pred_class_new,pred_class_k1_new=model3(images3d_new.to(device),k,v)
            pred_class_list_new=[]
            for i in range(len(labels1)):
                temp_pred_list_new=[]
                for t in range(len(pred_class_new[i])):
                    temp_pred_list_new.append(pred_class_new[i][t].item())
                pred_class_list_new.append(temp_pred_list_new)
            for i in range(len(labels1)):
                find=False
                f_ac=False
                for t in range(len(labels1[i])):
                    if labels1[i][t].item() in pred_class_list[i]:
                        if f_ac==False:
                            ac+=1
                            f_ac=True
                        for k in range(len(labels2_true[i])):
                            if labels2_true[i][k].item() in pred_class_list_new[i]:
                                accu_num+=1
                                find=True
                                break
                    if find==True:
                        break
            for i in range(len(labels1)):
                find=False
                for t in range(len(labels1[i])):
                    if labels1[i][t].item() == pred_class_k1[i][0].item():
                        for k in range(len(labels2_true[i])):
                            if labels2_true[i][k].item() == pred_class_k1_new[i].item():
                                acc_num1+=1
                                find=True
                                break
                    if find==True:
                        break
            loss2_1=loss_function1(pred_new1, labels2_loss_true1.to(device))
            loss2_1=scaler.scale(loss2_1)
            loss2=loss2_1
            loss2=scaler.scale(loss2)
            torch.nn.utils.clip_grad_norm_(parameters=model1.parameters(), max_norm=10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(parameters=model3.parameters(), max_norm=10, norm_type=2)
            accu_loss += loss.detach()
            accu_loss2+=loss2.detach()
            data_loader.desc = "[test {}] l:{:.3f},l2:{:.3f}, acc1:{:.3f},acc:{:.3f},acck1:{:.3f}".format(epoch,
                                                                        accu_loss.item()/(step + 1),
                                                                        accu_loss2.item() / (step + 1),
                                                                        ac/sample_num,
                                                                        accu_num / sample_num,
                                                                        acc_num1/sample_num
                                                                        )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

    return accu_loss.item() / (step + 1),accu_loss2.item()/(step + 1),ac/sample_num, accu_num / sample_num,acc_num1/sample_num