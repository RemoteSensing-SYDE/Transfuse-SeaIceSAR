import os
from tkinter import image_names
from PIL import Image
from matplotlib import patches
from sklearn.feature_extraction import image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2

# Added by No@
# -----------------------------
import itertools
import scipy.io as sio
from icecream import ic
from sklearn.preprocessing._data import _handle_zeros_in_scale
import joblib

def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, train_pixels=None):
    img[np.isnan(img)] = np.mean(img) # Filter NaN values.
    rows, cols, bands = img.shape

    if train_pixels is None:
        h = np.arange(0, rows)
        w = np.arange(0, cols)
        train_pixels = np.asarray(list(itertools.product(h, w))).transpose()

    min_value, max_value = [], []
    for band in range(bands):
        hist = np.histogram(img[train_pixels[0], train_pixels[1], band].ravel(), bins=bins) # select training pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        min_value.append(hist[1][len(cum_hist[cum_hist<bth])])
        max_value.append(hist[1][len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]

class Min_Max_Norm_Denorm():

    def __init__(self, img, mask, feature_range=[-1, 1]):

        self.feature_range = feature_range
        train_pixels = np.asarray(np.where(mask==0))

        self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.0005, uth=0.9995, train_pixels=train_pixels)
        img = self.median_filter(img.copy())
        
        self.min_val = np.nanmin(img[train_pixels[0], train_pixels[1]], axis=0)
        self.max_val = np.nanmax(img[train_pixels[0], train_pixels[1]], axis=0)
    
    def median_filter(self, img):
        kernel_size = 25
        outliers = (img < self.clips[0]) + (img > self.clips[1])
        out_idx = np.asarray(np.where(outliers))
        for i in range(out_idx.shape[1]):
            x = out_idx[0][i]
            y = out_idx[1][i]
            a = x - kernel_size//2 if x - kernel_size//2 >=0 else 0
            c = y - kernel_size//2 if y - kernel_size//2 >=0 else 0
            b = x + kernel_size//2 if x + kernel_size//2 <= img.shape[0] else img.shape[0]
            d = y + kernel_size//2 if y + kernel_size//2 <= img.shape[1] else img.shape[1]
            img[x, y] = np.median(img[a:b, c:d], axis=(0, 1))
        
        return img

    def clip_image(self, img):
        return np.clip(img.copy(), self.clips[0], self.clips[1])

    def Normalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale
        
        # img = self.clip_image(img.copy())
        img = self.median_filter(img.copy())
        img *= scale
        img += min_
        return img

    def Denormalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale

        img = img.copy() - min_
        img /= scale
        return img

def Split_Image(rows=5989, cols=2985, no_tiles_h=5, no_tiles_w=5):
    '''
    Split the image in tiles to define regions of training and validation
    for dense prediction approaches like semantic seegmentation

    returns mask (same size of the image).
    '''    

    xsz = rows // no_tiles_h
    ysz = cols // no_tiles_w

    # Tiles coordinates
    h = np.arange(0, rows, xsz)
    w = np.arange(0, cols, ysz)
    if (rows % no_tiles_h): h = h[:-1]
    if (cols % no_tiles_w): w = w[:-1]
    tiles = list(itertools.product(h, w))

    val_tiles = [1, 3, 7]          # Choose tiles by visual inspection
                                        # to guaratee all classes in both sets
                                        # (train and validation)

    mask = np.zeros((rows, cols))       # 0 Training Tiles
                                        # 1 Validation Tiles
    for i in val_tiles:
        t = tiles[i]
        finx = rows if (rows-(t[0] + xsz) < xsz) else (t[0] + xsz)
        finy = cols if (cols-(t[1] + ysz) < ysz) else (t[1] + ysz)
        # ic(finx-t[0], finy-t[1])
        mask[t[0]:finx, t[1]:finy] = 1
    
    return mask

def Split_in_Patches(rows, cols, patch_size, mask, lbl, percent=0, ref_r=0, ref_c=0):

    """
    Extract patches coordinates for each set, training, validation, and test

    Everything  in this function is made operating with
    the upper left corner of the patch

    (ref_r, ref_c)  Optional coordinates (patch upper-left corner) from which
                    we start the sliding window in all directions. This guarantees taking
                    the patch (ref_r : ref_r+patch_size, ref_c : ref_c+patch_size)
                    Useful to randomize the patch extraction
    """

    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    # overlap -= overlap % 2
    stride = patch_size - overlap

    # Add Padding to the image to match with the patch size
    lower_row_pad = (stride - ref_r        % stride) % stride
    upper_row_pad = (stride - (rows-ref_r) % stride) % stride
    lower_col_pad = (stride - ref_c        % stride) % stride
    upper_col_pad = (stride - (cols-ref_c) % stride) % stride
    pad_tuple_msk = ( (lower_row_pad, upper_row_pad+overlap), (lower_col_pad, upper_col_pad+overlap) )

    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')

    # Extract patches coordinates
    k1, k2 = (rows+lower_row_pad+upper_row_pad)//stride, (cols+lower_col_pad+upper_col_pad)//stride
    print('Total number of patches: %d x %d' %(k1, k2))
    print('Checking divisibility: %d , %d' %((rows+lower_row_pad+upper_row_pad)%stride, 
                                             (cols+lower_col_pad+upper_col_pad)%stride))

    train_mask, val_mask, test_mask = [np.zeros_like(mask_pad) for i in range(3)]
    train_mask[mask_pad==0] = 1
    val_mask  [mask_pad==1] = 1
    test_mask [mask_pad==2] = 1

    train_patches, val_patches, test_patches = [[] for i in range(3)]
    only_bck_patches = 0

    for i in range(k1):
        for j in range(k2):
            if lbl[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                # Train
                if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                    train_patches.append((i*stride, j*stride))
                # Val                 !!!!!Not necessary with high overlap!!!!!!!!
                elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                    val_patches.append((i*stride, j*stride))
                # Test                !!!!!Not necessary with high overlap!!!!!!!!
                elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                    test_patches.append((i*stride, j*stride))
            else:
                only_bck_patches += 1
    print('Background patches: %d' %(only_bck_patches))
    
    return train_patches, val_patches, test_patches, pad_tuple_msk

class RadarSAT2_Dataset():
    '''
    Load the dataset -> complete scene
    saves information related to classes like the color composition of segmented maps
    '''
    def __init__(self, image_root, gt_root, args, phase="train"):

        # Loading Data
        self.image = sio.loadmat(image_root)
        self.image = self.image[list(self.image)[-1]]

        self.gts  = sio.loadmat(gt_root)
        self.gts = self.gts[list(self.gts)[-1]].astype("float")
        self.gts -= 1                           # classes [0; n_clases-1]
                                                # background = -1        
        self.background = np.ones_like(self.gts)
        self.background[self.gts < 0] = 0       # Mask to cancel background pixels
                                                # background = 0
        self.gts[self.gts<0] = 0

        self.classes = ["Background", "Young ice", "First-year ice", "Multi-year ice", "Open water"]
        self.class_colors = np.uint8(np.array([[0, 0, 0],           # Background
                                               [170, 40, 240],      # Young ice
                                               [255, 255, 0],       # First year ice
                                               [255, 0, 0],         # Multi-year ice
                                               [150, 200, 255]]))    # Open water

        if phase == "train":
            self.batch_size = args.batch_size
            self.patch_size = args.patch_size
            self.patch_overlap = args.patch_overlap
            self.data_info_dir = args.data_info_dir
            os.makedirs(self.data_info_dir, exist_ok=True)
            
            # Data augmentation
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5),
                    A.HorizontalFlip(),
                    A.VerticalFlip()
                ]
            )

            self.define_sets()
    
    def define_sets(self):
        '''
        1st:    The scene is divided in tiles of equal size aproximatelly
                Some tiles are chossen for training and others for validation

        2nd:    Extract patches from each set (these are the images that feed the neural network)
        '''
        
        # Split image in Train and validation sets
        rows, cols = self.gts.shape
        no_tiles_h, no_tiles_w = 5, 3
        mask_train_val = Split_Image(rows=rows, cols=cols, no_tiles_h=no_tiles_h, no_tiles_w=no_tiles_w)
        Image.fromarray(np.uint8(mask_train_val*255)).save(self.data_info_dir + '/mask_tr_0_vl_1.tif')
        np.save(self.data_info_dir + '/mask_tr_0_vl_1', mask_train_val)
        
        # ========= NORMALIZE IMAGE =========
        self.norm = Min_Max_Norm_Denorm(self.image, mask_train_val)
        joblib.dump(self.norm, self.data_info_dir + '/norm_params.pkl')
        self.image = self.norm.Normalize(self.image)

        # Extract patches coordinates
        self.train_patches, \
        self.val_patches, _, self.pad_tuple = Split_in_Patches(rows, cols, self.patch_size, mask_train_val,
                                                               self.background, percent=self.patch_overlap)
        print("--------------")
        print("Training Patches:   %d"%(len(self.train_patches)))
        print("Validation Patches: %d"%(len(self.val_patches)))
        print("--------------")

        # Padding
        self.background = np.pad(self.background, self.pad_tuple, mode = 'symmetric')
        self.gts = np.pad(self.gts, self.pad_tuple, mode = 'symmetric')
        self.image = np.pad(self.image, self.pad_tuple+tuple([(0,0)]), mode = 'symmetric')
    
    def get_batch(self, stage="train"):
        
        '''
        generator that returns samples in batches 
        '''
        if stage == "train":
            samples = self.train_patches.copy()
            np.random.shuffle(samples)
        elif stage == "validation":
            samples = self.val_patches
        samples = np.array_split(samples, np.ceil(len(samples) / self.batch_size))

        for i in range(len(samples)):
            images, gts, bckg = [], [], []
            for x, y in samples[i]:
                im = self.image[x : x + self.patch_size, y : y + self.patch_size]
                gt = self.gts[x : x + self.patch_size, y : y + self.patch_size]
                bc = self.background[x : x + self.patch_size, y : y + self.patch_size]

                # Image.fromarray(np.uint8((im+self.norm.min_val)*255/(self.norm.max_val-self.norm.min_val))[:,:,:3])\
                #                                    .save(self.data_info_dir + '/im.tif')
                # Image.fromarray(np.uint8(gt*255/4)).save(self.data_info_dir + '/gt.tif')
                # Image.fromarray(np.uint8(bc*255  )).save(self.data_info_dir + '/bc.tif')

                transformed = self.transform(image=im, masks=[gt, bc])
                im = transformed['image']
                gt = transformed['masks'][0]
                bc = transformed['masks'][1]
                
                # Image.fromarray(np.uint8((im+self.norm.min_val)*255/(self.norm.max_val-self.norm.min_val))[:,:,:3])\
                #                                    .save(self.data_info_dir + '/im.tif')
                # Image.fromarray(np.uint8(gt*255/4)).save(self.data_info_dir + '/gt.tif')
                # Image.fromarray(np.uint8(bc*255  )).save(self.data_info_dir + '/bc.tif')

                images.append(im)
                gts.append(gt)
                bckg.append(bc)
            
            yield np.asarray(images), np.asarray(gts), np.asarray(bckg)

# -----------------------------


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SkinDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt



# if __name__ == '__main__':
#     path = 'data/'
#     tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

#     for i in range(50):
#         img, gt = tt.__getitem__(i)

#         img = torch.transpose(img, 0, 1)
#         img = torch.transpose(img, 1, 2)
#         img = img.numpy()
#         gt = gt.numpy()

#         plt.imshow(img)
#         plt.savefig('vis/'+str(i)+".jpg")
 
#         plt.imshow(gt[0])
#         plt.savefig('vis/'+str(i)+'_gt.jpg')
