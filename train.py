import os
import argparse
import torch
import torch.optim as optim
import random
import os,gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'      #set gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
gc.collect()
torch.cuda.empty_cache()    
from DataGenerator_others import MRIDataGenerator
#from DataGenerator_MRI import MRIDataGenerator          #This is for ADNI data generation
from vit_2d_sa import own_model as create_model
from vit_3d_ga1 import own_model as create_model1
from vit_3d_ga2 import own_model as create_model2
from utils import train_one_epoch, evaluate, test
import dill
warnings.filterwarnings('ignore')        #ignore warnings


def main(args):
    random.seed( 3407 )          #random seed setting
    model1 = create_model(has_logits=False)
    model2= create_model1(has_logits=False)
    model3= create_model2(has_logits=False)
    '''#Want to generate the data by yourself? Use the code in these lines.
    train_dataset=MRIDataGenerator(args.data_path,
                                     batchSize=args.batch_size* torch.cuda.device_count(),
                                     idx_fold=0,
                                     split='train')
    val_dataset=MRIDataGenerator(args.data_path,
                                     batchSize=args.batch_size* torch.cuda.device_count(),
                                     idx_fold=0,
                                     split='val')
    test_dataset=MRIDataGenerator(args.data_path,
                                     batchSize=args.batch_size* torch.cuda.device_count(),
                                     idx_fold=0,
                                     split='test')'''
    with open('./data_pkl/OrganMNIST/train.pkl','rb') as f:          #here we load the OrganMNIST as example
        train_dataset = dill.load(f)
    with open('./data_pkl/OrganMNIST/val.pkl','rb') as f:          #here we load the OrganMNIST as example
        val_dataset = dill.load(f)
    with open('./data_pkl/OrganMNIST/test.pkl','rb') as f:          #here we load the OrganMNIST as example
        test_dataset = dill.load(f)
    print(args.batch_size*torch.cuda.device_count())            # This is the total batch_size
    
    nw=0       #num workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    model1=model1.to('cuda')
    model2=model2.to('cuda')
    model3=model3.to('cuda')          #to gpu
    '''if args.weights != "":          #You can load weights here, but we default you have no pre-trained weights.
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict1 = torch.load(args.weights, map_location='gpu')      #weights for the first model
        weights_dict2=torch.load(args.weights2, map_location='gpu')      #weights for the second model
        weights_dict3=torch.load(args.weights3, map_location='gpu')      #weights for the third model
        model1.load_state_dict(weights_dict1, strict=True)
        model2.load_state_dict(weights_dict2, strict=True)
        model3.load_state_dict(weights_dict2, strict=True)
        del weights_dict1
        del weights_dict2
        del weights_dict3
        gc.collect()
        torch.cuda.empty_cache()
        model1=model1.to('cuda')
        model2=model2.to('cuda')
        model3=model3.to('cuda')'''

    pg = [p for p in model1.parameters() if p.requires_grad]+[q for q in model2.parameters() if q.requires_grad]
    pg1=[j for j in model3.parameters() if j.requires_grad]+[p for p in model1.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-3)
    optimizer2 = optim.AdamW(pg1, lr=args.lr, weight_decay=1e-3)

    for epoch in range(args.epochs):
        train_loss, train_loss1,train_high,train_acc,train_acc_k1,x_set = train_one_epoch(model1=model1,
                                                model2=model2,
                                                model3=model3,
                                                optimizer=optimizer,
                                                optimizer2=optimizer2,
                                                data_loader=train_loader,
                                                device=next(model1.parameters()).device,
                                                epoch=epoch)
        val_loss,val_loss1,val_high,val_acc,val_acc_k1= evaluate(model1=model1,
                                    model2=model2,
                                    model3=model3,
                                    data_loader=val_loader,
                                    device=next(model1.parameters()).device,
                                    epoch=epoch
                                    )   
        test_loss,test_loss1,test_high,test_acc,test_acc_k1=test(model1=model1,
                    model2=model2,
                    model3=model3,
                    data_loader=test_loader,
                    device=next(model1.parameters()).device,
                    epoch=epoch)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00005)

    # pretrain weights, default none
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--weights2', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--weights3', type=str, default='',
                        help='initial weights path')
    parser.add_argument('data_path', type=str, default='',
                        help='initial data path')

    opt = parser.parse_args()

    main(opt)
