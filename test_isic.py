# from tkinter import Image
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S
from utils.dataloader import test_dataset, RadarSAT2_Dataset
import imageio

from utils.utils import Image_reconstruction
from PIL import Image
from icecream import ic


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/TransFuse_S/TransFuse.pth')
    parser.add_argument('--Dataset_dir', type=str,
                        default='../../Dataset/RADARSAT-2-CP/', help='dataset_path')
    parser.add_argument('--test_path', type=str,
                        default='Scene56/', help='path to test dataset')
    parser.add_argument('--data_info_dir', type=str, default='./data_info')

    parser.add_argument('--save_path', type=str, default='./results/', help='path to save inference segmentation')
    parser.add_argument('--patch_size', type=int, default=320, help='patche size (square)')
    parser.add_argument('--patch_overlap', type=float, default=0.1, help='Overlap between patches')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')

    args = parser.parse_args()

    image_root = '{}/{}/all_bands.mat'.format(args.Dataset_dir, args.test_path)
    gt_root = '{}/{}/temp_labels/label_map.mat'.format(args.Dataset_dir, args.test_path)
    # test_loader = test_dataset(image_root, gt_root)
    test_data =  RadarSAT2_Dataset(image_root, gt_root, args, phase="test")
    
    # ========= NORMALIZE IMAGE =========
    norm_params = joblib.load(args.data_info_dir + '/norm_params.pkl')
    test_data.image = norm_params.Normalize(test_data.image)

    model = TransFuse_S(in_chans=test_data.image.shape[2], num_classes=args.n_classes).cuda()
    model.load_state_dict(torch.load(args.ckpt_path))
    model.cuda()
    model.eval()

    os.makedirs(args.save_path, exist_ok=True)

    print('evaluating model: ', args.ckpt_path)

    # Classifiying complete image
    pred_obj = Image_reconstruction(None, model, args.n_classes, 
                         patch_size=args.patch_size, overlap_percent=args.patch_overlap)
    probs = pred_obj.Inference(test_data.image)
    y_pred = np.argmax(probs, axis=0)

    # Colored prediction (class_colors[0] is background, so we use ypred+1 )
    colored_labels = test_data.class_colors[y_pred+1]

    # Metrics
    test_data.background = test_data.background[:y_pred.shape[0],:]
    test_data.gts = test_data.gts[:y_pred.shape[0],:]

    y_pred = y_pred[test_data.background == 1]
    y_true = test_data.gts[test_data.background == 1]

    acc = sum(y_pred == y_true) / len(y_pred)

    # Save results
    print("Test accuracy: {:.3f}%".format(acc*100))

    f = open(args.save_path + '/metrics.txt', 'a')
    f.write('------\n')
    f.write("Test accuracy: {:.3f}%\n".format(acc*100))
    f.write('------\n\n\n')
    f.close()

    np.save(args.save_path + '/output_probs.npy', probs)
    Image.fromarray(colored_labels).save(args.save_path + '/output_colored_labels.png')


    # dice_bank = []
    # iou_bank = []
    # acc_bank = []

    # for i in range(test_loader.size):
    #     image, gt = test_loader.load_data()
    #     gt = 1*(gt>0.5)
    #     image = image.cuda()

    #     with torch.no_grad():
    #         _, _, res = model(image)

    #     res = res.sigmoid().data.cpu().numpy().squeeze()
    #     res = 1*(res > 0.5)

    #     if opt.save_path is not None:
    #         imageio.imwrite(opt.save_path+'/'+str(i)+'_pred.jpg', res)
    #         imageio.imwrite(opt.save_path+'/'+str(i)+'_gt.jpg', gt)

    #     dice = mean_dice_np(gt, res)
    #     iou = mean_iou_np(gt, res)
    #     acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

    #     acc_bank.append(acc)
    #     dice_bank.append(dice)
    #     iou_bank.append(iou)

    # print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        # format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
