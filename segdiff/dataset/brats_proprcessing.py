'this file is for transfering 3D BRaTs MRI to 2D Slices of jpg image for training'
import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm


def nii2np_data(img_root, img_name, upper_per, lower_per, output_root=None, modality=None):
    # img_name = (img_path.split('/')[-1]).split('.')[0]
    modality = modality.split(',')

    img_file_label = os.path.join(img_root, img_name, img_name + '_' + 'seg' + '.nii.gz')
    img_label = nib.load(img_file_label)
    img_label = img_label.get_fdata()
    img_label = img_label.astype(np.uint8)    
    
    '''find the slice with maximum tumor area '''
    img_label = np.ones(img_label.shape) * (img_label > 0)  
    # np.sum(img_label[:,:,70:90], axis=(0,1))：对每个切片的像素值进行求和，得到长度为20的数组，表示每个切片的肿瘤面积
    if np.max(np.sum(img_label[:,:,70:90], axis=(0,1))) == 0:
        print(f'{img_name}没有肿瘤部分')
        return 
    
    else:
        slice = np.argmax(np.sum(img_label[:,:,70:90], axis=(0,1))) + 70
    
    '''generate image for each modality'''
    out = []
    for mod_num in range(len(modality)):
        img_file = os.path.join(img_root, img_name, img_name + '_' + modality[mod_num] + '.nii.gz')
        img = nib.load(img_file)
        img = (img.get_fdata())

        '''normalize the [lower_per, lower_per] of the brain to [-3,3]'''
        perc_upper = ((img > 0).sum() * (1 - upper_per)) / (img.shape[0] * img.shape[1] * img.shape[2])  
        perc_lower = ((img > 0).sum() * lower_per) / (img.shape[0] * img.shape[1] * img.shape[2])
        upper_value = np.percentile(img, (1 - perc_upper) * 100)
        lower_value = np.percentile(img, perc_lower * 100)
        img_half = (upper_value - lower_value) / 2
        img = (img - img_half) / (upper_value - lower_value) * 6  # normalize the bottom (1-x%) of the brain to [-3,3]
        # 获得切片
        img_slice = img[:, :, slice]
        out.append(img_slice)
        
    # 保存各个模态的slice文件
    dirs_all_mod = os.path.join(output_root, 'input')
    if not os.path.exists(dirs_all_mod):
        os.makedirs(dirs_all_mod) 
    # torch.stack(tensors, dim)函数用于沿一个新的维度将序列中的张量堆叠起来。如果没有指定dim参数，它将作为一个新的维度添加到张量的最前面
    # out = torch.stack(out)  # shape:(4,240,240)
    out = np.stack(out, axis=0)
    input_filename = os.path.join(dirs_all_mod, img_name + '_input_' + str(slice))
    np.save(input_filename, out)
    
    # 保存label
    dirs_seg = os.path.join(output_root, 'seg')
    if not os.path.exists(dirs_seg):
        os.makedirs(dirs_seg)
    filename_seg = os.path.join(dirs_seg, img_name + '_seg_' + str(slice))
    img_slice_seg = img_label[:, :, slice]
    np.save(filename_seg, img_slice_seg)

    
# def nii2np_data(img_root, img_name, upper_per, lower_per, output_root=None, modality=None):
#     # img_name = (img_path.split('/')[-1]).split('.')[0]
#     modality = modality.split(',')
    
    
#     '''generate image for each modality'''
#     for mod_num in range(len(modality)):
#         img_file = os.path.join(img_root, img_name, img_name + '_' + modality[mod_num] + '.nii.gz')
#         img = nib.load(img_file)
#         img = (img.get_fdata())
#         img_original = img

#         '''normalize the [lower_per, lower_per] of the brain to [-3,3]'''
#         perc_upper = ((img > 0).sum() * (1 - upper_per)) / (img.shape[0] * img.shape[1] * img.shape[
#             2])  # find the proportion of top (upper_per)% intensity of the brain within the whole 3D image
#         perc_lower = ((img > 0).sum() * lower_per) / (img.shape[0] * img.shape[1] * img.shape[2])
#         upper_value = np.percentile(img, (1 - perc_upper) * 100)
#         lower_value = np.percentile(img, perc_lower * 100)
#         img_half = (upper_value - lower_value) / 2
#         img = (img - img_half) / (upper_value - lower_value) * 6  # normalize the bottom (1-x%) of the brain to [-3,3]

#         img_file_label = os.path.join(img_root, img_name, img_name + '_' + 'seg' + '.nii.gz')
#         img_label = nib.load(img_file_label)
#         img_label = img_label.get_fdata()
#         img_label = img_label.astype(np.uint8)


#         '''find the slice with maximum tumor area '''
#         img_label = np.ones(img_label.shape) * (img_label > 0)
#         # np.sum(img_label[:,:,70:90], axis=(0,1))：对每个切片的像素值进行求和，得到长度为20的数组，表示每个切片的肿瘤面积
#         if np.max(np.sum(img_label[:,:,70:90], axis=(0,1))) == 0:  
#             print('pass')
#             pass
#         else:
#             slice = np.argmax(np.sum(img_label[:,:,70:90], axis=(0,1))) + 70
            
#             # 保存各个模态的切片
#             dirs_mod = os.path.join(output_root, modality[mod_num])
#             if not os.path.exists(dirs_mod):
#                 os.makedirs(dirs_mod)
#             filename = os.path.join(dirs_mod, img_name + '_' + modality[mod_num] + '_' + str(slice))
#             img_slice = img[:, :, slice]
#             np.save(filename, img_slice)

#             if mod_num == 0:
#                 # 保存label
#                 dirs_seg = os.path.join(output_root, 'seg')
#                 if not os.path.exists(dirs_seg):
#                     os.makedirs(dirs_seg)
#                 filename_seg = os.path.join(dirs_seg, img_name + '_seg_' + str(slice))
#                 img_slice_seg = img_label[:, :, slice]
#                 np.save(filename_seg, img_slice_seg)
                
#                 dirs_brainmask = os.path.join(output_root, 'brainmask')
#                 if not os.path.exists(dirs_brainmask):
#                     os.makedirs(dirs_brainmask)
#                 filename_brainmask = os.path.join(dirs_brainmask, img_name + '_brainmask_' + str(slice))
#                 img_brainmask = (img_original > 0).astype(int)
#                 img_slice_brainmask = img_brainmask[:, :, slice]
#                 np.save(filename_brainmask, img_slice_brainmask)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="the directory in which the data is stored", type=str, default='./autodl-tmp/data/BraTS2021/All_Data')
    parser.add_argument("--output_dir", help="the directory to store the preprocessed data", type=str, default='./autodl-tmp/data/BraTS2021/processed_data')
    parser.add_argument("--modality", help="The generated modality, like 't1', 't2', or 'flair'. Multi-modality separate by ',' without space, like 't1,t2'", type=str,
                        default='t1,t1ce,t2,flair')
    parser.add_argument("--upper_per", help="The upper percentage of brain area to be normalized, the value needs to be within [0-1], like 0.9", type=float, default=0.9)
    parser.add_argument("--lower_per", help="The lower percentage of brain area to be normalized, the value needs to be within [0-1], like 0.02", type=float, default=0.02)
    
    args = parser.parse_args()
    img_root = args.data_dir
    img_output_root = args.output_dir
    img_output_root_train = os.path.join(img_output_root, 'train')
    img_output_root_valid = os.path.join(img_output_root, 'valid')
    img_output_root_test = os.path.join(img_output_root, 'test')
    train_txt = './autodl-tmp/data/BraTS2021/brats_split_training.txt'
    valid_txt = './autodl-tmp/data/BraTS2021/brats_split_validing.txt'
    test_txt = './autodl-tmp/data/BraTS2021/brats_split_testing.txt'

    MOD = args.modality
    with open(train_txt) as file:
        for path in tqdm(file):
            # path[:-1]是去掉\n
            nii2np_data(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root=img_output_root_train, modality=MOD)

    with open(valid_txt) as file:
        for path in tqdm(file):
            nii2np_data(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root=img_output_root_valid, modality=MOD)
            
    with open(test_txt) as file:
        for path in tqdm(file):
            nii2np_data(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root=img_output_root_test, modality=MOD)