import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import datasets
from model import L2CS
from utils import select_device
import matplotlib.pyplot as plt
from utils import natural_keys, gazeto3d, angular
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--train_gazeMpiimage_dir', dest='train_gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--train_gazeMpiilabel_dir', dest='train_gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/train', type=str)
    parser.add_argument(
        '--val_gazeMpiimage_dir', dest='val_gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--val_gazeMpiilabel_dir', dest='val_gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/val', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=2, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output
    
    
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    
    
    if data_set=="gaze360":
        model, pre_url = getArch_weights(args.arch, 90)
        if args.snapshot == '':
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        else:
            saved_state_dict = torch.load(args.snapshot)
            model.load_state_dict(saved_state_dict)
        
        
        model.cuda(gpu)
        dataset=datasets.Gaze360(args.gaze360label_dir, args.gaze360image_dir, transformations, 180, 4)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
        

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)
       

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()
                
                iter_gaze += 1
                
                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )
        
          
            if epoch % 1 == 0 and epoch < num_epochs:
                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output +'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )
            

   
    elif data_set=="mpiigaze":
        train_folder = os.listdir(args.train_gazeMpiilabel_dir)
        val_folder = os.listdir(args.val_gazeMpiilabel_dir)
        train_folder.sort()
        val_folder.sort()
        trainlabelpathombined = [os.path.join(args.train_gazeMpiilabel_dir, j) for j in train_folder]
        vallabelpathombined = [os.path.join(args.val_gazeMpiilabel_dir, j) for j in val_folder]
        for fold in range(15):
            model, pre_url = getArch_weights(args.arch, 28)
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            #model = nn.DataParallel(model)
            model.to(gpu)
            print('Loading data.')
            dataset=datasets.Mpiigaze(trainlabelpathombined,args.train_gazeMpiimage_dir, transformations, True, fold)
            val_dataset=datasets.Mpiigaze(vallabelpathombined,args.val_gazeMpiimage_dir, transformations, True, fold)
            train_loader_gaze = DataLoader(
                dataset=dataset,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=4,
                pin_memory=True)
            val_loader_gaze = DataLoader(
                dataset=val_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
            torch.backends.cudnn.benchmark = True

            summary_name = '{}_{}'.format('L2CS-mpiigaze', int(time.time()))
            

            if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
                os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))

            
            criterion = nn.CrossEntropyLoss().cuda(gpu)
            reg_criterion = nn.MSELoss().cuda(gpu)
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(28)]
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

            # Optimizer gaze
            optimizer_gaze = torch.optim.Adam([
                {'params': get_ignored_params(model), 'lr': 0},
                {'params': get_non_ignored_params(model), 'lr': args.lr},
                {'params': get_fc_params(model), 'lr': args.lr}
            ], args.lr)

            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\n"
            print(configuration)
           
            train_loss_gaze = []
            train_loss_pitch = []
            val_loss_gaze = []
            val_loss_pitch = []
            train_avg_MAE = []
            val_avg_MAE = []
            for epoch in range(num_epochs):
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
                te_loss_pitch = te_loss_gaze = 0
                tr_loss_pitch = tr_loss_gaze = 0
                train_avg_error = val_avg_error = .0

                #训练
                for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)
                    

                    label_pitch = cont_labels_gaze[:,0].float()*np.pi/180
                    label_yaw = cont_labels_gaze[:,1].float()*np.pi/180

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)

                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1
                    tr_loss_pitch = sum_loss_pitch_gaze/iter_gaze
                    tr_loss_gaze = sum_loss_yaw_gaze/iter_gaze
                    
                    
                    pitch_predicted = pitch_predicted*np.pi/180
                    yaw_predicted = yaw_predicted*np.pi/180

                    for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                        p = p.data.cpu().numpy()
                        y = y.data.cpu().numpy() 
                        pl = pl.data.cpu().numpy()   
                        yl = yl.data.cpu().numpy()                      
                        train_avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        
                #验证
                model.eval()
                for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(val_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)
                    

                    label_pitch = cont_labels_gaze[:,0].float()*np.pi/180
                    label_yaw = cont_labels_gaze[:,1].float()*np.pi/180

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)

                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1

                    if (i + 1) % 100 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                              'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                  epoch + 1,
                                  num_epochs,
                                  i + 1,
                                  len(dataset) // batch_size,
                                  sum_loss_pitch_gaze / iter_gaze,
                                  sum_loss_yaw_gaze / iter_gaze
                              )
                              )
                    te_loss_pitch = sum_loss_pitch_gaze/iter_gaze
                    te_loss_gaze = sum_loss_yaw_gaze/iter_gaze
                    
                    pitch_predicted = pitch_predicted*np.pi/180
                    yaw_predicted = yaw_predicted*np.pi/180

                    for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                        p = p.data.cpu().numpy()
                        y = y.data.cpu().numpy() 
                        pl = pl.data.cpu().numpy()   
                        yl = yl.data.cpu().numpy()                      
                        val_avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))

                if  train_avg_error:
                    if (epoch+1) % 1 == 0:
                        print('Epoch [%d/%d], Losses: '
                            'train_avg_error %.4f,val_avg_error %.4f' % (
                                epoch+1,
                                num_epochs,
                                train_avg_error,
                                val_avg_error
                            )
                            )


                    train_loss_pitch.append(tr_loss_pitch.data.cpu().numpy())
                    train_loss_gaze.append(tr_loss_gaze.data.cpu().numpy())
                    val_loss_pitch.append(te_loss_pitch.data.cpu().numpy())
                    val_loss_gaze.append(te_loss_gaze.data.cpu().numpy())
                    train_avg_MAE.append(train_avg_error)
                    val_avg_MAE.append(val_avg_error)
            
            if train_avg_MAE:

                x = range(1,num_epochs+1)
                #plt.plot(x,l_gaze,"r",label='gaze_yaw')
                #plt.plot(x,l_pitch,"b",label='gaze_pitch')
                plt.title('The avg_loss of '+ 'fold' + str(fold), fontsize=20)
                plt.plot(x,train_avg_MAE,"r",label='train_loss')
                plt.plot(x,val_avg_MAE,"b",label='val_loss')
                plt.xlabel('epoches', fontsize=20)
                plt.ylabel('Loss', fontsize=20)
                plt.legend(loc='upper right')
                plt.grid()
                plt.savefig('./loss/'+ 'avg_loss_' +str(fold) +'.jpg')
                plt.clf()

                plt.title('The gaze_loss of '+ 'fold' + str(fold), fontsize=20)
                plt.plot(x,train_loss_pitch,"r",label='train_loss_pitch')
                plt.plot(x,val_loss_pitch,"y",label='val_loss_pitch')
                plt.plot(x,train_loss_gaze,"b",label='train_loss_gaze')
                plt.plot(x,val_loss_gaze,"g",label='val_loss_gaze')
                plt.xlabel('epoches', fontsize=20)
                plt.ylabel('Loss', fontsize=20)
                plt.legend(loc='upper right')
                plt.grid()
                plt.savefig('./loss/'+ 'gaze_loss_' +str(fold) +'.jpg')
                plt.clf()
                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    output+ summary_name+'/fold' + str(fold) +'/'+
                                    '_epoch_' + str(epoch+1) + '.pth.tar')
                        )

                    
                    

   
