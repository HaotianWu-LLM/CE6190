"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tqdm import tqdm

from guided_diffusion import dist_util, logger
# from guided_diffusion.bratsloader import BRATSDataset
from dataset.bratsloader2021 import BRATSDataset3D
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    # 对于b个结果，在相同位置求均值
    return th.sum(a, 0, keepdim=True) / b

def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            # 逐元素相乘，s和mvres应该shape一样
            r = s * mvres
            res = r if i == 0 else th.cat((res,r),0)
        nres = mv(res)
        # 逐元素相减
        gap = th.mean(th.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    tran_list = [transforms.Resize((args.image_size,args.image_size)),]
    transform_test = transforms.Compose(tran_list)
    ds = BRATSDataset3D(args.data_dir,'test',transforms=transform_test)
    # ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  
    
    for _ in tqdm(range(len(data))):
        b, m, path = next(data)
        # b, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        # print(img.shape)
        slice_ID=path[0].split('.')[0]
        print('slice_ID:',slice_ID)
        # slice_ID=path[0].split("/", -1)[3]

        # viz.image(visualize(img[0,0,...]), opts=dict(caption="img input0"))
        # viz.image(visualize(img[0, 1, ...]), opts=dict(caption="img input1"))
        # viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        # viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        # viz.image(visualize(img[0, 4, ...]), opts=dict(caption="img input4"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            # print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            s = th.tensor(sample)
            # print(s.shape)
            enslist.append(s[:,-1,:,:])
           # th.save(s, './outputs/'+str(slice_ID)+'_output'+str(i)) #save the generated mask
            
        # 保存集成图像，staple是自己写的函数,type(ensres)=torch.Tensor，shape = [1,1,256,256]
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0) # torch.stack会增加一个维度 .squeeze(0)会将第一个维度消掉   
        vutils.save_image(ensres, fp = os.path.join('./outputs/', str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10) 

def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
