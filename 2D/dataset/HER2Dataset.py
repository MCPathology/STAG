import glob
import numpy as np
import pandas
import torch
import random
import os
import json
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import KFold

Image.MAX_IMAGE_PIXELS = None

class HER2Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path='./data/HER2/',
                 mode='train',
                 k_folds=5,
                 fold_index=0,
                 seed=42,
                 selected_genes=None,
                 patch_size=224,
                 split_save_path='./HER2_kfold_splits.json',
                 model_name = 'STAG'):
        if selected_genes is None:
            selected_genes_path = 'select_genes/HER2_Selected_Genes.npy'
            if not os.path.exists(selected_genes_path):
                raise FileNotFoundError(f"基因文件未找到: {selected_genes_path}")
            selected_genes = np.load(selected_genes_path, allow_pickle=True).tolist()

        self.verbose = False
        self.mode = mode
        self.patch_size = patch_size
        
        self.wsi_imgs, self.wsi_imgs2, self.wsi_imgs3 = [], [], []
        self.st_feats, self.st_spots_pixel_map = [], []
        
        self.transform = T.Compose([T.ToTensor()])
        
        all_splits = None
        if os.path.exists(split_save_path):
            print(f"--- 发现已存在的划分文件: {split_save_path}，正在加载... ---")
            with open(split_save_path, 'r') as f:
                all_splits = json.load(f)
            if all_splits.get('k_folds') != k_folds or all_splits.get('seed') != seed:
                print("--- [警告] 划分文件中的参数与当前设置不符，将重新生成划分。 ---")
                all_splits = None

        if all_splits is None:
            print(f"--- 未找到或参数不匹配，正在生成新的 {k_folds}-折 划分... ---")
            image_path = os.path.join(path, 'images/HE/')
            all_wsi_files = np.array(sorted(glob.glob(os.path.join(image_path, '*.jpg'))))
            if len(all_wsi_files) == 0:
                raise FileNotFoundError(f"在路径 '{image_path}' 下没有找到任何 .jpg 文件。")

            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            
            all_splits = {'k_folds': k_folds, 'seed': seed, 'total_files': len(all_wsi_files), 'folds': []}

            for i, (train_indices, val_indices) in enumerate(kf.split(all_wsi_files)):
                train_files = [os.path.basename(f) for f in all_wsi_files[train_indices]]
                val_files = [os.path.basename(f) for f in all_wsi_files[val_indices]]
                all_splits['folds'].append({'fold_index': i, 'train_files': train_files, 'val_files': val_files})
            
            with open(split_save_path, 'w') as f:
                json.dump(all_splits, f, indent=4)
            print(f"--- 新的划分已保存到: {split_save_path} ---")

        if fold_index >= len(all_splits['folds']):
            raise ValueError(f"fold_index ({fold_index}) 超出范围。")
        
        current_fold_info = all_splits['folds'][fold_index]
        
        if self.mode == 'train':
            wsi_filenames = [os.path.join(path, 'images/HE', f) for f in current_fold_info['train_files']]
            print(f"--- [Fold {fold_index+1}/{k_folds}, Mode: Train] 加载 {len(wsi_filenames)} 个训练文件 ---")
        elif self.mode == 'val':
            wsi_filenames = [os.path.join(path, 'images/HE', f) for f in current_fold_info['val_files']]
            print(f"--- [Fold {fold_index+1}/{k_folds}, Mode: Val] 加载 {len(wsi_filenames)} 个验证文件 ---")
        else:
            raise ValueError(f"模式 '{self.mode}' 无效。")

        for wsi_file in wsi_filenames:
            print('正在处理:', wsi_file)
            wsi_img = Image.open(wsi_file)

            wsi_basename = os.path.basename(wsi_file).split('.')[0]
            st_feats_path = os.path.join(path, 'count-matrices', f'{wsi_basename}.tsv')
            st_spots_pixel_map_path = os.path.join(path, 'spot-selection', f'{wsi_basename}_selection.tsv')
            
            if not (os.path.exists(st_feats_path) and os.path.exists(st_spots_pixel_map_path)):
                print(f"    [警告] 找不到对应的特征或坐标文件，跳过: {wsi_basename}")
                continue

            feats_all = pandas.read_csv(st_feats_path, sep='\t', index_col=0, header=0)
            spots_pixel_map_all = pandas.read_csv(st_spots_pixel_map_path, sep='\t', header=0)

            for _, spot_info in spots_pixel_map_all.iterrows():
                idx = f"{int(spot_info['x'])}x{int(spot_info['y'])}"
                center_x, center_y = spot_info['pixel_x'], spot_info['pixel_y']
                
                crop_size_1, crop_size_2, crop_size_3 = self.patch_size, self.patch_size * 2, self.patch_size * 4

                try:
                    def get_patch_and_resize(wsi_img, center_x, center_y, crop_s, target_s):
                        x1, y1 = int(center_x - crop_s // 2), int(center_y - crop_s // 2)
                        x2, y2 = int(center_x + crop_s // 2), int(center_y + crop_s // 2)
                        img_w, img_h = wsi_img.size
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
                        patch = wsi_img.crop((x1, y1, x2, y2))
                        if patch.size != (target_s, target_s):
                            patch = patch.resize((target_s, target_s), Image.BILINEAR)
                        return patch

                    patch1 = get_patch_and_resize(wsi_img, center_x, center_y, crop_size_1, self.patch_size)
                    patch2 = get_patch_and_resize(wsi_img, center_x, center_y, crop_size_2, self.patch_size)
                    patch3 = get_patch_and_resize(wsi_img, center_x, center_y, crop_size_3, self.patch_size)

                except Exception as e:
                    if self.verbose: print(f'裁剪图像时出错: {e}, spot: {idx}')
                    continue

                try:
                    feats = np.array(feats_all.loc[idx][selected_genes])
                except KeyError:
                    if self.verbose: print(f'基因表达数据中找不到 spot: {idx}')
                    continue
                
                spot_sum = np.sum(feats)
                if spot_sum == 0:
                    if self.verbose: print(f'spot 的基因总和为0，跳过: {idx}')
                    continue
                
                feats = np.log1p(feats * 1e6 / spot_sum)
                
                self.st_spots_pixel_map.append([idx, center_x, center_y])
                self.st_feats.append(np.array(feats))
                self.wsi_imgs.append(patch1)
                self.wsi_imgs2.append(patch2)
                self.wsi_imgs3.append(patch3)

        print(f'--- [Fold {fold_index+1}/{k_folds}, Mode: {self.mode}] 加载完成，总样本数: {len(self.st_spots_pixel_map)} ---')
        
    def __len__(self):
        return len(self.st_feats)

    def augmentation(self, img, img2, img3):
        if random.random() < 0.5:
            img, img2, img3 = img.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT), img3.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img, img2, img3 = img.transpose(Image.FLIP_TOP_BOTTOM), img2.transpose(Image.FLIP_TOP_BOTTOM), img3.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            degree = random.randint(0, 360)
            img, img2, img3 = T.RandomRotation((degree, degree), fill=255)(img), T.RandomRotation((degree, degree), fill=255)(img2), T.RandomRotation((degree, degree), fill=255)(img3)
        return img, img2, img3
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img, img2, img3 = self.augmentation(self.wsi_imgs[index], self.wsi_imgs2[index], self.wsi_imgs3[index])
        else:
            img, img2, img3 = self.wsi_imgs[index], self.wsi_imgs2[index], self.wsi_imgs3[index]
        return self.st_spots_pixel_map[index], self.transform(img), self.transform(img2), self.transform(img3), self.st_feats[index]
