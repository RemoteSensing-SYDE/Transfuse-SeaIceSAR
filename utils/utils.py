import torch
import numpy as np
from icecream import ic


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):])).item()


class Image_reconstruction(object):

    '''
    This class performes a slide windows for dense predictions.
    If we consider overlap between consecutive patches then we will keep the central part 
    of the patch (stride, stride)

    considering a small overlap is usually useful because dense predections tend 
    to be unaccurate at the border of the image
    '''
    def __init__ (self, inputs, model, output_c_dim, patch_size=256, overlap_percent=0):

        self.inputs = inputs
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.model = model
    
    def Inference(self, tile):
        
        '''
        Normalize before calling this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
        
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')
        tile_pad = torch.from_numpy(tile_pad.transpose((2, 0, 1))).float().cuda()

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((self.output_c_dim, k1*stride, k2*stride), dtype='float32')

        for i in range(k1):
            for j in range(k2):
                
                patch = tile_pad[:, i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size)]
                # patch = patch[np.newaxis,...]
                patch = patch.unsqueeze(0)

                _, _, infer = self.model(patch)
                infer = infer.detach().cpu().numpy()

                probs[:, i*stride : i*stride+stride, 
                         j*stride : j*stride+stride] = infer[0, :, overlap//2 : overlap//2 + stride, 
                                                                   overlap//2 : overlap//2 + stride]
            print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[:, :k1*stride-step_row, :k2*stride-step_col]

        return probs
