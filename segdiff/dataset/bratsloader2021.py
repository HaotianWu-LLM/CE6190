import torch
import torch.nn
import numpy as np
import os
import nibabel
from os.path import join
from pathlib import Path
from torch.utils.data import Dataset



class BRATSDataset3D(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod='input', transforms=None):
        super(BRATSDataset3D, self).__init__()

        assert mode in ['train','valid','test']
        self.data_root = data_root
        self.mode = mode  # 是做训练，验证还是测试
        self.input_mod = input_mod # 输入数据的模态

        self.transforms = transforms
        self.case_names_input = sorted(list(Path(os.path.join(self.data_root, self.mode,input_mod)).iterdir()))
        self.case_names_seg = sorted(list(Path(os.path.join(self.data_root, self.mode,'seg')).iterdir()))
        # print(os.getcwd()) # /root/MedSegDiff
        

    def __getitem__(self, index: int) -> tuple:
        name_input = self.case_names_input[index].name
        name_seg = self.case_names_seg[index].name
        
        base_dir_input = join(self.data_root, self.mode, self.input_mod, name_input)
        base_dir_seg = join(self.data_root, self.mode, 'seg', name_seg)
        
        image = np.load(base_dir_input)
        image = torch.from_numpy(image)  # dtype=torch.float64
        
        label = np.load(base_dir_seg) # 只有0和1
        label = torch.from_numpy(label).unsqueeze(0)   # dtype=torch.float64
        
        if self.transforms:
            state = torch.get_rng_state()
            image = self.transforms(image) # shape:(4,256,256) 包括't1', 't1ce', 't2', 'flair'
            torch.set_rng_state(state) 
            label = self.transforms(label) # shape:(1,256,256)
        
        return (image, label, name_input) # virtual path
    
    

    def __len__(self):
        return len(self.case_names_input)
    
    
    
    
    
