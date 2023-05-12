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
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

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
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            bbx1 = np.random.randint(0, W-cut_w)
            bby1 = np.random.randint(0, H-cut_h)
            bbx2 = bbx1+cut_w
            bby2 = bby1+cut_h
        
        elif method == 'randomShape': 
            scale = np.random.beta(beta, beta)

            while True:
                ratio = np.random.uniform(0.3, 1/(0.3))

                w_rat, h_rat = np.sqrt(scale*ratio), np.sqrt(scale/ratio)

                cut_w = int(W * w_rat)
                cut_h = int(H * h_rat)

                if W-cut_w > 0 and H-cut_h > 0:
                    break

            bbx1 = np.random.randint(0, W-cut_w)
            bby1 = np.random.randint(0, H-cut_h)
            bbx2 = bbx1+cut_w
            bby2 = bby1+cut_h
        
        elif method == 'fixedCenter':
            ratio = np.random.beta(beta, beta)
            cut_rat = np.sqrt(ratio)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = int(W/2)
            cy = int(H/2)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
                
        elif method == 'vertex':
            ratio = np.random.beta(beta, beta)
            cut_rat = np.sqrt(ratio)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

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

def calc_ins_std_mean(x):
    """extract feature map statistics"""
    size = x.size()
    N, C = size[:2]
    
    if len(size) == 4:
        std, mean = torch.std_mean(x, dim=(2,3), keepdim=True)
    elif len(size) == 3:
        std, mean = torch.std_mean(x, dim=2, keepdim=True)
    else:
        raise NotImplementedError('Unsupported feature shape.')
    
    return std.view(N, C, 1, 1), mean.view(N, C, 1, 1)


class WindowNorm2d(nn.Module):
    bboxs = None
    bboxs_len = 0

    """ WindowNorm"""
    def __init__(self, num_features, mask_thres=0.7, eps=1e-5,  alpha=0.1, mix=True, grid=False, input_size=224, mask_patch_size=32, affine=False, cached=True, device='cuda'):
        super(WindowNorm2d, self).__init__()

        if cached and  WindowNorm2d.bboxs is None:
            WindowNorm2d.bboxs = np.load('./bboxs.npy') # mask_thres = 0.7
            WindowNorm2d.bboxs_len = len(WindowNorm2d.bboxs)

        self.num_features = num_features
        self.mask_thres= mask_thres
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size

        self.eps = eps
        self.device = device

        # mixup
        self.mix = mix
        self.alpha = torch.tensor(alpha, device='cuda')
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

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
            N, C, H, W = x.size()

            if self.grid:
                # with grid mask 
                if hasattr(self, 'generator'):
                    pass
                else:
                    scale = round(self.input_size/H)
                    self.generator = MaskGenerator(input_size=self.input_size, mask_patch_size=self.mask_patch_size, model_patch_size=scale, mask_ratio=self.mask_thres)

                masked_x, _ = self.generator(x)

            else:
                if WindowNorm2d.bboxs is None:
                    bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=1, bbx_thres=self.mask_thres)
                else:
                    bbx1, bby1, bbx2, bby2 = WindowNorm2d.bboxs[np.random.randint(0, WindowNorm2d.bboxs_len)]

                    bbx1, bby1 = int(W*bbx1), int(H*bby1)
                    bbx2, bby2 = int(W*bbx2), int(H*bby2)

                masked_x = torch.narrow(torch.narrow(x, 2, bbx1, bbx2-bbx1), 3, bby1, bby2-bby1)

            std, mean = calc_ins_std_mean(masked_x)


            if self.mix and not self.grid:
                N, C, _, _ = x.shape

                lmda = self.beta.sample((N, C, 1, 1))

                global_std, global_mean = calc_ins_std_mean(x)

                mean = mean*lmda + global_mean*(1-lmda)
                std = std*lmda + global_std*(1-lmda)

        else:
            std, mean = calc_ins_std_mean(x)

        normalized_feat = (x-mean) / (std+self.eps)


        if self.affine:
            return self.weight[..., None, None]*normalized_feat + self.bias[..., None, None]
        else:
            return normalized_feat
    
    def __repr__(self):
        return "WindowNorm2d(num_features={}, mask_thres={}, alpha={}, mix={}, grid={}, input_size={}, mask_patch_size={} , eps={}, affine={}, cached={}, device={:s})".format(self.num_features, self.mask_thres, self.alpha, self.mix, self.grid, self.input_size, self.mask_patch_size, self.eps, self.affine, WindowNorm2d.bboxs is not None, self.device)

    @classmethod
    def convert_WIN_model(cls, module, mask_thres=0.7, alpha=0.1, mix=True, grid=False, input_size=224, mask_patch_size=32, affine=False, cached=False, device='cuda'):
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

            mod = WindowNorm2d(module.num_features, mask_thres=mask_thres,  alpha=alpha, mix=mix, grid=grid, input_size=input_size, mask_patch_size=mask_patch_size, affine=affine, cached=cached, device=device)

        for name, child in module.named_children():
                mod.add_module(
                    name, cls.convert_WIN_model(child, mask_thres=mask_thres,  alpha=alpha, mix=mix, grid=grid, input_size=input_size, mask_patch_size=mask_patch_size, affine=affine, cached=cached, device=device)
                )

        del module
        return mod

    @classmethod
    def convert_IN_model(cls, module):
        '''
        Recursively traverse module and its children to replace all instances of
        ``BatchNorm2d`` with :class:`InstanceNorm`.

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

    @classmethod
    def convert_Identity_model(cls, module):
        '''
        Recursively traverse module and its children to replace all instances of
        ``BatchNorm2d`` with :class:`Identity`.

        Args:
            module (torch.nn.Module): input module
        Example::
            >>> # model is an instance of torch.nn.Module
            >>> sync_bn_model = convert_Identity_model(model)
        '''
        mod = module

        if isinstance(module, torch.nn.BatchNorm2d):
            mod = nn.Identity()

        for name, child in module.named_children():
                mod.add_module(
                    name, cls.convert_Identity_model(child)
                )

        del module
        return mod

    @classmethod
    def convert_GN_model(cls, module):
        '''
        Recursively traverse module and its children to replace all instances of
        ``BatchNorm2d`` with :class:`GroupNorm`.

        Args:
            module (torch.nn.Module): input module
        Example::
            >>> # model is an instance of torch.nn.Module
            >>> sync_bn_model = convert_GN_model(model)
        '''
        mod = module

        if isinstance(module, torch.nn.BatchNorm2d):
            mod = nn.GroupNorm(64, module.num_features) 

        for name, child in module.named_children():
                mod.add_module(
                    name, cls.convert_GN_model(child)
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
        mask_idx = torch.randperm(self.token_count)[:self.mask_count]

        mask = torch.zeros(self.token_count, dtype=int)

        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat_interleave(self.scale, dim=0).repeat_interleave(self.scale, dim=1)

        size = x.size()
        N, C = size[:2]

        x = x.contiguous().view(N, C, -1)

        x = x[:,:, mask.contiguous().view(-1).bool()]

        return x, mask