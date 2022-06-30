# -*- coding: utf-8 -*-
'''
@Date          : 2022-03-30 22:26:37
@Author        : joe1chief
@Contact       : joe1chief1993@gmail.com
@Copyright     : 2022 SJTU IMR
@License       : CC BY-NC-SA 4.0
@Last Modified : joe1chief 2022-03-30 22:26:37
@Des           : None

@Log           : None

'''
import torch
from torch import nn

import numpy as np

def cn_rand_bbox(size, beta, bbx_thres, method='original'):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        if method == 'original':
            ratio = np.random.beta(beta, beta)
            cut_rat = np.sqrt(ratio)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        elif method == 'fixedShape':
            ratio = np.random.beta(beta, beta)
            cut_rat = np.sqrt(ratio)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            bbx1 = np.random.randint(0, W-cut_w)
            bby1 = np.random.randint(0, H-cut_h)
            bbx2 = bbx1+cut_w
            bby2 = bby1+cut_h
        
        elif method == 'randomShape': 
            scale = np.random.beta(beta, beta)

            while True:
                ratio = np.random.uniform(0.3, 1/(0.3))

                w_rat, h_rat = np.sqrt(scale*ratio), np.sqrt(scale/ratio)

                cut_w = np.int(W * w_rat)
                cut_h = np.int(H * h_rat)

                if W-cut_w > 0 and H-cut_h > 0:
                    break

            bbx1 = np.random.randint(0, W-cut_w)
            bby1 = np.random.randint(0, H-cut_h)
            bbx2 = bbx1+cut_w
            bby2 = bby1+cut_h
        
        elif method == 'fixedCenter':
            ratio = np.random.beta(beta, beta)
            cut_rat = np.sqrt(ratio)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            cx = int(W/2)
            cy = int(H/2)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
                
        elif method == 'vertex':
            ratio = np.random.beta(beta, beta)
            cut_rat = np.sqrt(ratio)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            bbx1 = np.random.choice([0, W-cut_w])
            bby1 = np.random.choice([0, H-cut_h])
            bbx2 = bbx1+cut_w
            bby2 = bby1+cut_h
        
        else:
            raise error('%s has not implemented !!!' % method)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)

        if ratio >= bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2

def calc_ins_mean_var(x):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    assert (len(size) == 4) or (len(size) == 3)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2, unbiased=False).view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, var

class WindowNorm2d(nn.Module):
    bboxs = None
    bboxs_len = 0

    """ WindowNorm"""
    def __init__(self, num_features, mask_thres=0.7, eps=1e-5,  alpha=0.1, mix=True, grid=False, input_size=224, mask_patch_size=32, affine=False, cached=True):
        super(WindowNorm2d, self).__init__()

        if cached and  WindowNorm2d.bboxs is None:
            WindowNorm2d.bboxs = np.load('./bboxs.npy') # mask_thres = 0.7
            WindowNorm2d.bboxs_len = len(WindowNorm2d.bboxs)

        self.num_features = num_features
        self.mask_thres= mask_thres
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size

        self.eps = eps

        # mixup
        self.mix = mix
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)

        # grid mask
        self.grid = grid

        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.training:
            if self.grid:
                # with grid mask 
                if hasattr(self, 'generator'):
                    pass
                else:
                    _, _, w, _ = x.shape
                    scale = round(self.input_size/w)
                    self.generator = MaskGenerator(input_size=self.input_size, mask_patch_size=self.mask_patch_size, model_patch_size=scale, mask_ratio=self.mask_thres)

                masked_x, _ = self.generator(x)

            else:
                if WindowNorm2d.bboxs is None:
                    bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=1, bbx_thres=self.mask_thres)
                else:
                    _, _, H, W = x.size()

                    bbx1, bby1, bbx2, bby2 = WindowNorm2d.bboxs[np.random.randint(0, WindowNorm2d.bboxs_len)]

                    # print(bbx1, bby1, bbx2, bby2)

                    bbx1, bby1 = int(W*bbx1), int(H*bby1)
                    bbx2, bby2 = int(W*bbx2), int(H*bby2)

                masked_x = x[:, :, bbx1:bbx2, bby1:bby2]

            mean, var = calc_ins_mean_var(masked_x)

            if self.mix:
                global_mean, global_var = calc_ins_mean_var(x)

                N, C, _, _ = x.shape

                lmda = self.beta.sample((N, C, 1, 1))
                lmda = lmda.to(x.device)

                mean = mean*lmda + global_mean * (1-lmda)
                var = var*lmda + global_var * (1-lmda)
        else:
            mean, var = calc_ins_mean_var(x)

        size = x.size()
        normalized_feat = (x - mean.expand(size)) / ((var+self.eps).sqrt().expand(size))

        if self.affine:
            return self.weight[..., None, None]*normalized_feat + self.bias[..., None, None]
        else:
            return normalized_feat
    
    def __repr__(self):
        return "WindowNorm2d(num_features={}, mask_thres={}, alpha={}, mix={}, grid={}, input_size={}, mask_patch_size={} , eps={}, affine={})".format(self.num_features, self.mask_thres, self.alpha, self.mix, self.grid, self.input_size, self.mask_patch_size, self.eps, self.affine)

    @classmethod
    def convert_WIN_model(cls, module, mask_thres=0.7,  alpha=0.1, mix=True, grid=False, input_size=224, mask_patch_size=32, affine=False, cached=False):
        '''
        Recursively traverse module and its children to replace all instances of
        ``BatchNorm2d`` with :class:`WindowNorm2d`.

        Args:
            module (torch.nn.Module): input module
        Example::
            >>> # model is an instance of torch.nn.Module
            >>> sync_bn_model = convert_WIN_model(model)
        '''
        mod = module

        if isinstance(module, torch.nn.BatchNorm2d):
            print(module)
            mod = WindowNorm2d(module.num_features, mask_thres=mask_thres,  alpha=alpha, mix=mix, grid=grid, input_size=input_size, mask_patch_size=mask_patch_size, affine=affine, cached=cached)

        for name, child in module.named_children():
                mod.add_module(
                    name, cls.convert_WIN_model(child, mask_thres=mask_thres,  alpha=alpha, mix=mix, grid=grid, input_size=input_size, mask_patch_size=mask_patch_size, affine=affine, cached=cached)
                )

        del module
        return mod

    @classmethod
    def convert_IN_model(cls, module):
        '''
        Recursively traverse module and its children to replace all instances of
        ``BatchNorm2d`` with :class:`InstanceNorm_patch`.

        Args:
            module (torch.nn.Module): input module
        Example::
            >>> # model is an instance of torch.nn.Module
            >>> sync_bn_model = convert_IN_model(model)
        '''
        mod = module

        if isinstance(module, torch.nn.BatchNorm2d):
            mod = nn.InstanceNorm2d(module.num_features)

        for name, child in module.named_children():
                mod.add_module(
                    name, cls.convert_IN_model(child)
                )

        del module
        return mod

class MaskGenerator:
    '''
        SIMMIM
    '''
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=2, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self, x):
        # mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        # mask_count = np.random.randint(self.mask_count, self.token_count)
        # print(mask_count)
        mask_idx = torch.randperm(self.token_count)[:self.mask_count]

        # print('mask_idx', mask_idx)

        mask = torch.zeros(self.token_count, dtype=int)

        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat_interleave(self.scale, dim=0).repeat_interleave(self.scale, dim=1)

        size = x.size()
        N, C = size[:2]

        x = x.contiguous().view(N, C, -1)

        x = x[:,:, mask.contiguous().view(-1).bool()]

        return x, mask





if __name__ == '__main__':
    import torchvision
    from torch.nn import BatchNorm2d

    import  torchvision.models as models
    # x = torch.ones(8, 1, 224, 224)

    # # norm = RecalibratedInstanceNorm_patch(64, affine=True).to('cuda:0')

    # # x = norm(x)

    # for i in range(8):
    #     bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=1, bbx_thres=0.7, method='fixedRatio')

    #     x[i,:, bbx1:bbx2, bby1:bby2] *= 0.6

    # torchvision.utils.save_image(x, 'fixedRatio_patch_mask.png')

    # bboxs = []

    # for i in range(10000000):

    #     bbx1, bby1, bbx2, bby2 = cn_rand_bbox([1, 3, 224, 224], beta=1, bbx_thres=0.7)

    #     print(bbx1, bby1, bbx2, bby2)

    #     bbox = [bbx1 / 224.0, bby1 / 224.0, bbx2 / 224.0, bby2 / 224.0]

    #     bboxs.append(bbox)

    # bboxs = np.array(bboxs)

    # np.save('bboxs.npy', bboxs)




    x = torch.rand(8, 16, 224, 224, device='cuda:1')


    norm = WindowNorm2d(16).to('cuda:1')


    x = norm(x)


    # norm = BatchNorm2d(16).to('cuda:1')

    # print(norm)


    # norm = InstanceNorm_patch.convert_PIN_model(norm)

    # print(norm)

    # x = norm(x)

    # resnet18 = models.resnet18().to('cuda:1')

    # print(resnet18)

    # resnet18_PIN = InstanceNorm_patch.convert_PIN_model(resnet18)

    # print(resnet18)

    # x = torch.rand(8, 3, 224, 224, device='cuda:1')


    # x = resnet18(x)