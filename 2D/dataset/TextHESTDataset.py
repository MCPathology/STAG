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
        raise FileNotFoundError(f"在目录 {st_path} 中未找到任何 .h5ad 文件。")
    return [osp.basename(f).replace('.h5ad', '') for f in ad_filenames]

def get_full_kfold_splits(data_root_path: str, n_splits: int = 5, random_state: int = 1553) -> List[Dict]:
    subset_names = get_all_subset_names(data_root_path)
    N = len(subset_names)
    print(f"--- 样本总数: {N}，将进行 {n_splits}-Fold 交叉验证 ---")
    if N < n_splits: 
        n_splits = N
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kfold_iterator = []
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(subset_names)):
        train_subsets = [subset_names[i] for i in train_indices]
        val_subsets = [subset_names[i] for i in val_indices]
        print(f"Fold {fold_idx}: 训练集样本数 = {len(train_subsets)}, 验证集样本数 = {len(val_subsets)}")
        kfold_iterator.append({'fold': fold_idx, 'train': train_subsets, 'val': val_subsets})
    return kfold_iterator


def get_patch_from_wsi(slide_obj, center_coords, crop_size, target_s):
    center_x, center_y = center_coords
    
    x1_wsi = int(center_x - crop_size // 2)
    y1_wsi = int(center_y - crop_size // 2)
    
    patch_pil_raw = slide_obj.read_region((x1_wsi, y1_wsi), 0, (crop_size, crop_size))
    if patch_pil_raw.mode == 'RGBA':
        patch_pil_raw = patch_pil_raw.convert('RGB')
        
    if patch_pil_raw.size != (target_s, target_s):
        patch_pil_raw = patch_pil_raw.resize((target_s, target_s), Image.BILINEAR)
        
    return patch_pil_raw


class HESTDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_path: str, 
                 mode: str, 
                 specific_subsets: List[str], 
                 selected_genes: List[str] = None, 
                 selected_genes_file_path: str = None,
                 text_encoding_file_path: str = None,
                 patch_size=224):

        self.data_root = data_root_path
        self.mode = mode
        self.patch_size = patch_size
        self.crop_size = patch_size * 2

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
        
        self.gene_text_encoding = None
        if text_encoding_file_path:
            if osp.exists(text_encoding_file_path):
                print(f"加载基因文本编码: {text_encoding_file_path}")
                text_encoding_np = np.load(text_encoding_file_path, allow_pickle=True)
                self.gene_text_encoding = torch.from_numpy(text_encoding_np).float()
                
                if self.selected_genes and self.gene_text_encoding.shape[0] != len(self.selected_genes):
                    print(f"⚠️ 警告: 基因列表数量 ({len(self.selected_genes)}) 与文本编码第一维度 ({self.gene_text_encoding.shape[0]}) 不匹配。")
                if self.gene_text_encoding.shape[1] != 768:
                    print(f"⚠️ 警告: 文本编码第二维度为 {self.gene_text_encoding.shape[1]}，预期为 768。")

            else:
                print(f"⚠️ 致命错误: 未找到文本编码文件 {text_encoding_file_path}。程序将退出。")
                import sys
                sys.exit()
        else:
            print("⚠️ 致命错误: 未提供 `text_encoding_file_path`。程序将退出。")
            import sys
            sys.exit()
        
        self.all_spot_data = [] 
        self.h5_cache = {} 
        self.wsi_cache = {}
        self.subset_files = specific_subsets 
        
        if not self.subset_files:
            print(f"⚠️ 警告: {self.mode.upper()} 模式没有分配到样本，跳过加载。")
        else:
            self._build_metadata_list_with_neighbors()

        self.transform = T.Compose([T.ToTensor()])
        
        print(f"\n✅ {self.mode.upper()} 模式加载完成。总 Spot 样本数: {len(self)}")
        
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
            if f: f.close()
        self.h5_cache.clear()
        for slide in self.wsi_cache.values():
            if slide: slide.close()
        self.wsi_cache.clear()

    def _build_metadata_list_with_neighbors(self):
        total_spots_count = 0
        
        for subset_name in self.subset_files:
            print(f"  - 正在为样本构建元数据: {subset_name}")
            
            ad_file = osp.join(self.data_root, 'st', f'{subset_name}.h5ad')
            wsi_file = osp.join(self.data_root, 'wsis', f'{subset_name}.tif')

            if not all(osp.exists(f) for f in [ad_file, wsi_file]):
                print(f"    跳过: 缺少 {subset_name} 的 .h5ad 或 .tif 文件。")
                continue
                
            adata = anndata.read_h5ad(ad_file)
            
            slide = self._get_wsi_file(wsi_file)
            if slide is None: continue

            if 'pxl_col_in_fullres' in adata.obs.columns and 'pxl_row_in_fullres' in adata.obs.columns:
                all_barcodes = adata.obs_names
                all_coords = adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values.astype(np.float32)
            else:
                print(f"    跳过: {subset_name} 的 .h5ad 文件中缺少坐标信息。")
                continue
            
            sample_gene_to_idx_map = {gene: i for i, gene in enumerate(adata.var_names)}
            target_indices_in_sample = [sample_gene_to_idx_map.get(gene, -1) for gene in self.selected_genes]
            
            if issparse(adata.X):
                X = adata.X.toarray()
            else:
                X = np.array(adata.X)

            current_wsi_spots = []
            for i, barcode in enumerate(all_barcodes):
                raw_expression_vector = X[i]
                ordered_vector = np.zeros(len(self.selected_genes), dtype=np.float32)

                for target_idx, source_idx in enumerate(target_indices_in_sample):
                    if source_idx != -1:
                        ordered_vector[target_idx] = raw_expression_vector[source_idx]
                
                spot_sum_filtered = np.sum(ordered_vector)
                if spot_sum_filtered == 0:
                    continue
                
                normalized_feats = np.log1p(ordered_vector * 1e6 / spot_sum_filtered).astype(np.float32)
                
                current_wsi_spots.append({
                    'id': f'{subset_name}_{barcode}', 
                    'coords': all_coords[i], 
                    'feats': normalized_feats,
                    'index_in_adata': i
                })
            
            if not current_wsi_spots: continue
            
            all_uniform_coords = np.array([s['coords'] for s in current_wsi_spots])
            
            
            for i in range(len(current_wsi_spots)):
                target_spot = current_wsi_spots[i]
                target_coords = target_spot['coords']
                
                distances = np.linalg.norm(all_uniform_coords - np.array(target_coords), axis=1)
                num_neighbors_to_find = min(9, len(current_wsi_spots))
                nearest_indices = np.argsort(distances)[:num_neighbors_to_find]
                
                patches_9 = []
                feats_9 = []
                
                for neighbor_idx_in_current_wsi in nearest_indices:
                    neighbor_spot = current_wsi_spots[neighbor_idx_in_current_wsi]
                    neighbor_coords = neighbor_spot['coords']
                    
                    try:
                        patch_pil = get_patch_from_wsi(
                            slide, neighbor_coords, self.crop_size, self.patch_size
                        )
                        patches_9.append(patch_pil)
                        feats_9.append(neighbor_spot['feats'])
                    except Exception as e:
                        print(f"    [警告] 邻居裁剪失败: {neighbor_spot['id']}, 错误: {e}")
                        patches_9 = []
                        feats_9 = []
                        break
                
                if patches_9:
                    num_found = len(patches_9)
                    if num_found < 9:
                        padding_patch = Image.new('RGB', (self.patch_size, self.patch_size), (255, 255, 255))
                        patches_9.extend([padding_patch] * (9 - num_found))
                        
                        num_genes = len(target_spot['feats'])
                        padding_feats = np.zeros(num_genes, dtype=np.float32)
                        feats_9.extend([padding_feats] * (9 - num_found))

                    self.all_spot_data.append({
                        'spot_info': [target_spot['id'], target_coords[0], target_coords[1]],
                        'img_center': patches_9[0],
                        'feats_center': feats_9[0],
                        'hypergraph_x': patches_9,
                        'hypergraph_x_exp': np.array(feats_9)
                    })
                    total_spots_count += 1
            
        print(f'--- 加载完成，总样本数: {total_spots_count} ---')


    def __len__(self):
        return len(self.all_spot_data)
    
    def augmentation(self, img):
        if self.mode != 'train':
            return img
        
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img= img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            degree = random.randint(0, 360)
            img = img.rotate(degree, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
        return img
    
    def __getitem__(self, index):
        data = self.all_spot_data[index]
        
        img_center_pil = data['img_center']
        img_center_pil = self.augmentation(img_center_pil)
        img_center_tensor = self.transform(img_center_pil)
        
        feats_center_tensor = torch.from_numpy(data['feats_center']).float()
        
        hypergraph_x_tensors = []
        for i, patch_pil in enumerate(data['hypergraph_x']):
            if i == 0: 
                hypergraph_x_tensors.append(img_center_tensor)
            else: 
                hypergraph_x_tensors.append(self.transform(patch_pil))

        hypergraph_x = torch.stack(hypergraph_x_tensors)
        hypergraph_x_exp = torch.from_numpy(data['hypergraph_x_exp']).float()
        gene_text_encoding_tensor = self.gene_text_encoding
        
        return (data['spot_info'], 
                img_center_tensor, 
                feats_center_tensor, 
                hypergraph_x, 
                hypergraph_x_exp,
                gene_text_encoding_tensor)
