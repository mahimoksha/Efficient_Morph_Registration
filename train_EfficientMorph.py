from comet_ml import Experiment
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
import timeit
from models.EfficientMorph import CONFIGS as CONFIGS_TM
import models.EfficientMorph as EfficientMorph
from models.EfficientMorph_hires import CONFIGS as CONFIGS_TM_HIRES
import models.EfficientMorph_hires as EfficientMorph_hires
from torchsummary import summary
import socket
import time
import argparse
experiment = Experiment(
    api_key="put-api-key-here",
    project_name="oasis-experiments-training-without-segloss",
    workspace="efficientmorph",
)
experiment.add_tag("EfficientMorph_2x3_variant_patch_size_2")
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(args):
    batch_size = 1
    train_dir = 'OASIS/OASIS_L2R_2021_task03/All/'
    val_dir = 'OASIS/OASIS_L2R_2021_task03/Test/'
    weights = [1, 1, 1] # loss weights
    save_dir = 'EfficientMorph_2x3_variant_patch_size_2/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0005 # learning rate
    epoch_start = 0
    # max_epoch = 500 #max traning epoch
    max_epoch = 200
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    if args.hi_res:
        config = CONFIGS_TM_HIRES['EfficientMorph_2x3_2_hires']
        model = EfficientMorph_hires.EfficientMorph(config)
        model.cuda()
    else:
        config = CONFIGS_TM['EfficientMorph_2x3_2']
        model = EfficientMorph.EfficientMorph(config)
        model.cuda()

    summary(model,(2,160, 192, 224))
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(trainable_params)


    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''

    if cont_training:
        print("Continuing Traning")
        epoch_start = 50
        model_dir = 'experiments/Epoch_50/'+save_dir
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        optimizer = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['optimizer']
        updated_lr = get_lr(optimizer=optimizer)
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr
    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    if args.dataset_name == "IXI":
        atlas_dir = 'Path_to_IXI_data/atlas.pkl'
        if args.seg_loss:
            train_set = datasets.IXIBrainInferDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
        else:
            train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
        val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    elif args.dataset_name == "OASIS":
        train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
        val_set = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    else:
        train_set = datasets.L2RLUMIRJSONDataset(base_dir=train_dir, json_path=train_dir+'ReMIND2Reg_dataset.json', stage='train', transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    T_max = 20
    eta_min = 0.00005
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    criterion_ncc = losses.NCC_gauss()
    criterion_dsc = losses.DiceLoss()
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
    training_times = []
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        start_time = time.time()
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            if args.seg_loss:
                x_seg = data[2]
                y_seg = data[3]

                x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
                x_seg_oh = torch.squeeze(x_seg_oh, 1)
                x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

            x_in = torch.cat((x,y), dim=1)
            output, flow = model(x_in)
            if args.seg_loss:
                def_segs = []
                for i in range(36):
                    def_seg = model.spatial_trans(x_seg_oh[:, i:i + 1, ...].float(), flow.float())
                    def_segs.append(def_seg)
                def_seg = torch.cat(def_segs, dim=1)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_dsc = criterion_dsc(def_seg, y_seg.long()) * weights[1] if args.seg_loss else 0
            loss_reg = criterion_reg(flow, y) * weights[2]
            loss = loss_ncc  + loss_reg + loss_dsc
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del  x_in, loss

            if args.seg_loss:
                y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=36)
                y_seg_oh = torch.squeeze(y_seg_oh, 1)
                y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

            y_in = torch.cat((y, x), dim=1)
            output, flow = model(y_in)
            if args.seg_loss:
                def_segs = []
                for i in range(36):
                    def_seg = model.spatial_trans(y_seg_oh[:, i:i + 1, ...].float(), flow.float())
                    def_segs.append(def_seg)
                def_seg = torch.cat(def_segs, dim=1)
            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_dsc = criterion_dsc(def_seg, x_seg.long()) * weights[1] if args.seg_loss else 0
            loss_reg = criterion_reg(flow, x) * weights[2]
            loss = loss_ncc + loss_reg + loss_dsc
            loss_all.update(loss.item(), x.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del y_in
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, DSC: {:.6f},  Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_ncc.item(),
                                                                                                loss_dsc.item(),
                                                                                                loss_reg.item()))

        scheduler.step()
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        metrics = {'train_loss': loss_all.avg}
        experiment.log_metrics(metrics, step=epoch)
        loss_all.reset()
        training_times.append(time.time() - start_time)
        '''
        Validation
        '''
        if args.evaluation:
            eval_dsc = utils.AverageMeter()
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    data = [t.cuda() for t in data]
                    x = data[0]
                    y = data[1]
                    x_seg = data[2]
                    y_seg = data[3]
                    x_in = torch.cat((x, y), dim=1)
                    grid_img = mk_grid_img(8, 1, config.img_size)
                    output = model(x_in)
                    def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                    def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                    dsc = utils.dice_val_VOI(def_out.long(), y_seg.long(), dataset_name=args.dataset_name)
                    eval_dsc.update(dsc.item(), x.size(0))
                    print(eval_dsc.avg)
            best_dsc = max(eval_dsc.avg, best_dsc)
            if((epoch+1)%10==0):
                save_dir = 'Epoch_'+str(epoch+1)+'/'+'EfficientMorph_2x3_variant_patch_size_2/'
                if not os.path.exists('experiments/'+save_dir):
                    os.makedirs('experiments/'+save_dir)
                print('model saved to path '+'experiments/'+save_dir)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir='experiments/'+save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
            metrics = {'val_dice': eval_dsc.avg}
            experiment.log_metrics(metrics, step=epoch)
            del def_out, def_grid, grid_img, output
        
        

    experiment.log_parameters(
        {
            "parameters": trainable_params,
            "lr":lr,
            "scheduler":"cosine",
            "training_time": sum(training_times) / len(training_times)
        }
    )
    experiment.end()

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='EfficientMorph Training Script')
    parser.add_argument('--dataset_name', type=str, choices=['IXI', 'OASIS', 'Remind2Reg'], required=True,
                        help='Dataset name to use for training and validation.')
    parser.add_argument('--evaluation', type=bool, default=False,
                        help='Set to True to enable evaluation mode. Default is False.')
    parser.add_argument('--seg_loss', type=bool, default=False,
                        help='Set to True to include segmentation loss. Default is False.')
    parser.add_argument('--hi_res', type=bool, default=False,
                        help='Set to True to include segmentation loss. Default is False.')

    args = parser.parse_args()

    '''
    GPU configuration
    '''
    start = timeit.default_timer()
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main(args)
    stop = timeit.default_timer()
    print('Time: ', stop - start)   