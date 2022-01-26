import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S
from utils.dataloader import get_loader, test_dataset, RadarSAT2_Dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os


def structure_loss_old(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # weit is weight image that gives more importance to boundaries of the segments
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

# Added by No@
def structure_loss(pred, mask, bckg):
    # bckg cancel background pixels
    loss = bckg * CrossEntropyLoss(reduction='none')(pred, mask)
    return loss.mean()


# def train(train_loader, model, optimizer, epoch, best_loss):
def train(train_data, model, optimizer, epoch, best_loss, to_torch=True):

    save_path = 'snapshots/{}/'.format(args.train_save)
    os.makedirs(save_path, exist_ok=True)
    f = open(save_path + "Log.txt", "a")

    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    # accum = 0
    total_step = len(train_data.train_patches) // train_data.batch_size
    total_step += (len(train_data.train_patches) % train_data.batch_size) > 0
    
    # for i, pack in enumerate(train_loader, start=1):
    for i, pack in enumerate(train_data.get_batch(stage="train")):

        # ---- data prepare ----
        images, gts, bckg = pack
        if to_torch:
            images  = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
            gts     = torch.from_numpy(gts).int()
            bckg    = torch.from_numpy(bckg).int()
        if torch.cuda.is_available():
            images  = images.cuda()
            gts     = gts.cuda()
            bckg    = bckg.cuda()

        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts, bckg)
        loss3 = structure_loss(lateral_map_3, gts, bckg)
        loss2 = structure_loss(lateral_map_2, gts, bckg)

        loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

        if epoch:
            # ---- backward ----
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, args.batchsize)
        loss_record3.update(loss3.data, args.batchsize)
        loss_record4.update(loss4.data, args.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step or epoch == 0:
            print_line = 'Tr {} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]\n'\
                         .format(datetime.now(), epoch, args.epoch, i, total_step, loss_record2.show(), loss_record3.show(), loss_record4.show())
            print (print_line)
        if epoch == 0: break

    # Save Logs
    f.write(print_line)

    # Validation
    meanloss = validation(model, train_data)
    print_line = 'Vl {} Epoch [{:03d}], [meanloss: {:.4f}]\n'.format(datetime.now(), epoch, meanloss)
    print (print_line)
    f.write(print_line)
    
    if meanloss < best_loss:
        best_loss = meanloss
        torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth'% epoch)
        f.write("===== model saved =====")
        best_model_flag = 1
    else:
        best_model_flag = 0
    
    f.close()

    return best_loss, best_model_flag



def validation(model, data, to_torch=True):
    model.eval()
    loss_bank = []
    acc_bank = []
    for i, pack in enumerate(data.get_batch(stage="validation")):

        # ---- data prepare ----
        images, gts, bckg = pack
        if to_torch:
            images  = torch.from_numpy(images).float()
            gts     = torch.from_numpy(gts).int()
            bckg    = torch.from_numpy(bckg).int()
        if torch.cuda.is_available():
            images  = images.cuda()
            gts     = gts.cuda()
            bckg    = bckg.cuda()

        with torch.no_grad():
            _, _, res = model(images)
        loss = structure_loss(res, gts, bckg)

        res = torch.argmax(res, dim=1).detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        bckg = bckg.detach().cpu().numpy()

        res = res[bckg == 1]
        gts = gts[bckg == 1]

        acc = sum(res == gts) / len(res)

        loss_bank.append(loss.item())
        acc_bank.append(acc)
        
    print('Loss: {:.4f}, Acc: {:.4f}'.format(np.mean(loss_bank), np.mean(acc_bank)))

    return np.mean(loss_bank)

# def test(model, path):
#     model.eval()
#     mean_loss = []

#     for s in ['val', 'test']:
#         image_root = '{}/data_{}.npy'.format(path, s)
#         gt_root = '{}/mask_{}.npy'.format(path, s)
#         test_loader = test_dataset(image_root, gt_root)

#         dice_bank = []
#         iou_bank = []
#         loss_bank = []
#         acc_bank = []

#         for i in range(test_loader.size):
#             image, gt = test_loader.load_data()
#             image = image.cuda()

#             with torch.no_grad():
#                 _, _, res = model(image)
#             loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             gt = 1*(gt>0.5)            
#             res = 1*(res > 0.5)

#             dice = mean_dice_np(gt, res)
#             iou = mean_iou_np(gt, res)
#             acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

#             loss_bank.append(loss.item())
#             dice_bank.append(dice)
#             iou_bank.append(iou)
#             acc_bank.append(acc)
            
#         print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
#             format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

#         mean_loss.append(np.mean(loss_bank))

#     return mean_loss[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs after no improvements (stop criteria)')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--patch_size', type=int, default=320, help='patche size (square)')
    parser.add_argument('--patch_overlap', type=float, default=0.1, help='Overlap between patches')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')

    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    args = parser.parse_args()

    image_root = '{}/data_train.npy'.format(args.train_path)
    gt_root = '{}/mask_train.npy'.format(args.train_path)
    # train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize)
    # total_step = len(train_loader)
    train_data =  RadarSAT2_Dataset(image_root, gt_root, args)

    # ---- build models ----
    model = TransFuse_S(in_chans=train_data.image.shape[3], num_classes=args.n_classes).cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.lr, betas=(args.beta1, args.beta2))
    

    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    stop_criteria = args.patience
    for epoch in range(0, args.epoch + 1):
        best_loss, best_model_flag = train(train_data, model, optimizer, epoch, best_loss)

        if best_model_flag:
            stop_criteria = args.patience
        else: stop_criteria -= 1
        if not stop_criteria: break
