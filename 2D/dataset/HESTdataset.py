import anndata
import h5py
import os
import os.path as osp
import numpy as np
import torch
import random
import glob
import openslide
from PIL import Image
import torchvision.transforms as T
from scipy.sparse import issparse
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple

Image.MAX_IMAGE_PIXELS = None 


def get_all_subset_names(data_root_path: str) -> List[str]:
    st_path = osp.join(data_root_path, 'st')
    ad_filenames = sorted(glob.glob(st_path + '/*.h5ad'))
    if not ad_filenames:
        raise FileNotFoundError(f"在目录 {st_path} 中未找到任何 .h5ad 文件。请检查路径是否正确。")
    return [osp.basename(f).replace('.h5ad', '') for f in ad_filenames]

def get_full_kfold_splits(data_root_path: str, n_splits: int = 5, random_state: int = 1553) -> List[Dict]:
    subset_names = get_all_subset_names(data_root_path)
    N = len(subset_names)
    
    print(f"--- 样本总数: {N}，将进行 {n_splits}-Fold 交叉验证 ---")

    if N < n_splits: 
        print(f"警告：样本总数({N})小于K值({n_splits})，K值已自动调整为{N}。")
        n_splits = N
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    kfold_iterator = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(subset_names)):
        
        train_subsets = [subset_names[i] for i in train_indices]
        val_subsets = [subset_names[i] for i in val_indices]
        
        print(f"Fold {fold_idx}: 训练集样本数 = {len(train_subsets)}, 验证集样本数 = {len(val_subsets)}")
        
        kfold_iterator.append({
            'fold': fold_idx, 
            'train': train_subsets, 
            'val': val_subsets
        })
        
    return kfold_iterator


class HESTDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_path: str, 
                 mode: str, 
                 specific_subsets: List[str], 
                 selected_genes: List[str] = None, 
                 selected_genes_file_path: str = None,
                 patch_size=224):

        self.data_root = data_root_path
        self.mode = mode
        self.patch_size = patch_size

        if selected_genes is None and selected_genes_file_path:
            if osp.exists(selected_genes_file_path):
                print(f"加载预设基因列表: {selected_genes_file_path}")
                self.selected_genes = np.load(selected_genes_file_path, allow_pickle=True).tolist()
            else:
                print(f"⚠️ 警告: 未找到基因文件 {selected_genes_file_path}。")
                import sys
                sys.exit()
                self.selected_genes = None
        else:
            self.selected_genes = selected_genes
        
        self.spot_metadata = [] 
        self.h5_cache = {} 
        self.wsi_cache = {}
        self.subset_files = specific_subsets 
        
        if not self.subset_files:
            print(f"⚠️ 警告: {self.mode.upper()} 模式没有分配到样本，跳过加载。")
        else:
            self._build_metadata_list()

        self.transform = T.Compose([T.ToTensor()])
        
        print(f"\n✅ {self.mode.upper()} 模式加载完成。总 Spot 样本数: {len(self)}")
    
    def _build_metadata_list(self):
        for subset_name in self.subset_files:
            print(f"  - 正在为样本构建元数据: {subset_name}")
            
            ad_file = osp.join(self.data_root, 'st', f'{subset_name}.h5ad')
            patch_file = osp.join(self.data_root, 'patches', f'{subset_name}.h5')
            wsi_file = osp.join(self.data_root, 'wsis', f'{subset_name}.tif')

            if not all(osp.exists(f) for f in [ad_file, patch_file, wsi_file]):
                print(f"    跳过: 缺少 {subset_name} 的 .h5, .h5ad, 或 .tif 文件。")
                continue

            try:
                with h5py.File(patch_file, 'r') as f:
                    if isinstance(f['barcode'][0, 0], bytes):
                        patch_barcodes = [b.decode('utf-8') for b in f['barcode'][:, 0]]
                    else:
                        patch_barcodes = [b for b in f['barcode'][:, 0]]
            except Exception as e:
                print(f"    致命错误: 读取 {patch_file} 的 barcode 时出错: {e}")
                continue
            
            adata = anndata.read_h5ad(ad_file)
            try:
                adata_filtered = adata[patch_barcodes, :].copy()
            except KeyError:
                print(f"    跳过: {subset_name} 的 .h5 和 .h5ad 文件中的 barcodes 不完全匹配。")
                continue

            coords_source = None
            if 'pxl_col_in_fullres' in adata_filtered.obs.columns and 'pxl_row_in_fullres' in adata_filtered.obs.columns:
                print(f"    坐标来源: .h5ad 文件 (.obs)")
                patch_coords = adata_filtered.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
                coords_source = '.h5ad'
            else:
                try:
                    with h5py.File(patch_file, 'r') as f:
                        if 'coords' in f:
                            print(f"    坐标来源: .h5 文件")
                            patch_coords = f['coords'][:]
                            coords_source = '.h5'
                except Exception as e:
                    print(f"    致命错误: 尝试从 {patch_file} 回退读取坐标时出错: {e}")
                    continue
            patch_coords = patch_coords.astype(np.float32)
            
            if coords_source is None:
                print(f"    致命错误: 在 {subset_name} 的 .h5ad 和 .h5 文件中都找不到可用的坐标信息。")
                continue


            sample_gene_to_idx_map = {gene: i for i, gene in enumerate(adata_filtered.var_names)}

            target_indices_in_sample = [sample_gene_to_idx_map.get(gene, -1) for gene in self.selected_genes]

            if issparse(adata_filtered.X):
                X = adata_filtered.X.toarray()
            else:
                X = np.array(adata_filtered.X)

            for i, barcode in enumerate(patch_barcodes):
                raw_expression_vector = X[i]

                ordered_vector = np.zeros(len(self.selected_genes), dtype=np.float32)

                for target_idx, source_idx in enumerate(target_indices_in_sample):
                    if source_idx != -1:
                        ordered_vector[target_idx] = raw_expression_vector[source_idx]
                

                spot_sum_filtered = np.sum(ordered_vector)
                if spot_sum_filtered == 0:
                    continue
                
                normalized_feats = np.log1p(ordered_vector * 1e6 / spot_sum_filtered)
                expression_tensor = torch.FloatTensor(normalized_feats)
                
                self.spot_metadata.append({
                    'patch_file': patch_file,
                    'patch_index': i,
                    'wsi_file': wsi_file,
                    'global_barcode': f'{subset_name}_{barcode}',
                    'spot_coords': patch_coords[i],
                    'expression_tensor': expression_tensor 
                })
    
    def __len__(self):
        return len(self.spot_metadata)
    
    def _get_h5_file(self, path):
        if path not in self.h5_cache:
            self.h5_cache[path] = h5py.File(path, 'r')
        return self.h5_cache[path]

    def _get_wsi_file(self, path):
        if path not in self.wsi_cache:
            try:
                self.wsi_cache[path] = openslide.open_slide(path)
            except openslide.OpenSlideError as e:
                print(f"致命错误: OpenSlide 无法打开文件 {path}: {e}")
                self.wsi_cache[path] = None 
        return self.wsi_cache[path]

    def close_caches(self):
        for f in self.h5_cache.values():
            f.close()
        self.h5_cache.clear()
        for slide in self.wsi_cache.values():
            if slide:
                slide.close()
        self.wsi_cache.clear()

    def _get_multiscale_patch(self, slide_obj, center_coords, crop_size):
        center_x, center_y = center_coords
        top_left_x = int(center_x - crop_size // 2)
        top_left_y = int(center_y - crop_size // 2)
        patch = slide_obj.read_region((top_left_x, top_left_y), 0, (crop_size, crop_size))
        if patch.mode == 'RGBA':
            patch = patch.convert('RGB')
        return patch

    def __getitem__(self, index):
        metadata = self.spot_metadata[index]
        
        try:
            h5_file = self._get_h5_file(metadata['patch_file'])
            patch1_array = h5_file['img'][metadata['patch_index']]
            patch1_img = Image.fromarray(patch1_array)
        except Exception as e:
            print(f"错误：无法从 {metadata['patch_file']} 加载 patch 索引 {metadata['patch_index']}: {e}")
            return None

        slide = self._get_wsi_file(metadata['wsi_file'])
        if slide is None:
            return None 
        center_coords = metadata['spot_coords']

        patch2_img_raw = self._get_multiscale_patch(slide, center_coords, self.patch_size * 2 * 2)
        patch3_img_raw = self._get_multiscale_patch(slide, center_coords, self.patch_size * 2 * 4)
        
        patch2_img = patch2_img_raw.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        patch3_img = patch3_img_raw.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        if self.mode == 'train':
            do_hflip, do_vflip, do_rotate = random.random() < 0.5, random.random() < 0.5, random.random() < 0.5
            degree = random.randint(0, 360) if do_rotate else 0
            temp_imgs = []
            for img in [patch1_img, patch2_img, patch3_img]:
                if do_hflip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if do_vflip: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                if do_rotate: img = T.RandomRotation((degree, degree), fill=255)(img)
                temp_imgs.append(img)
            patch1_img, patch2_img, patch3_img = temp_imgs

        tensor1 = self.transform(patch1_img)
        tensor2 = self.transform(patch2_img)
        tensor3 = self.transform(patch3_img)

        spot_info = (metadata['global_barcode'], metadata['spot_coords'])
        processed_feats = metadata['expression_tensor']
        
        return spot_info, tensor1, tensor2, tensor3, processed_feats
