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

class cSCCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path='./data/GSE144240',
                 mode='train',
                 k_folds=5,
                 fold_index=0,
                 seed=42,
                 selected_genes=None,
                 patch_size=224,
                 split_save_path='./cSCC_kfold_splits.json',
                 text_path = 'select_genes/cSCC_bert_text_encode.npy',
                 num_neighbors=5,
                 geneformer_emb_dir=None
                 ): 
        if selected_genes is None:
            selected_genes_path = 'select_genes/cSCC_Selected_Genes.npy'
            if not os.path.exists(selected_genes_path):
                raise FileNotFoundError(f"基因文件未找到: {selected_genes_path}")
            selected_genes = np.load(selected_genes_path, allow_pickle=True).tolist()

        self.verbose = True
        self.mode = mode
        self.patch_size = patch_size
        self.text_path = text_path
        self.num_neighbors = num_neighbors
        
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"基因编码文件未找到: {self.text_path}")
        
        self.gene_text_encoding = np.load(self.text_path)
        
        if self.gene_text_encoding.shape != (250, 768):
            print(f"--- [警告] 加载的基因编码维度为 {self.gene_text_encoding.shape}，预期为 (250, 768)。 ---")
        self.gene_text_encoding = torch.from_numpy(self.gene_text_encoding).float()

        self.geneformer_emb_dir = geneformer_emb_dir
        self.geneformer_lookup = {}
        self.geneformer_dim = 0
        if geneformer_emb_dir is not None:
            self._load_geneformer_embeddings(geneformer_emb_dir)

        self.all_spot_data = []
        
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
            all_wsi_files = np.array(sorted(glob.glob(os.path.join(path, '*.jpg'))))
            if len(all_wsi_files) == 0:
                raise FileNotFoundError(f"在路径 '{path}' 下没有找到任何 .jpg 文件。")

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
            wsi_filenames = [os.path.join(path, f) for f in current_fold_info['train_files']]
            print(f"--- [Fold {fold_index+1}/{k_folds}, Mode: Train] 加载 {len(wsi_filenames)} 个训练文件 ---")
        elif self.mode == 'val':
            wsi_filenames = [os.path.join(path, f) for f in current_fold_info['val_files']]
            print(f"--- [Fold {fold_index+1}/{k_folds}, Mode: Val] 加载 {len(wsi_filenames)} 个验证文件 ---")
        else:
            raise ValueError(f"模式 '{self.mode}' 无效。")

        def get_patch_and_resize(wsi_img, center_x, center_y, crop_s, target_s):
            x1, y1 = int(center_x - crop_s // 2), int(center_y - crop_s // 2)
            x2, y2 = int(center_x + crop_s // 2), int(center_y + crop_s // 2)
            img_w, img_h = wsi_img.size
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            patch = wsi_img.crop((x1, y1, x2, y2))
            
            if patch.size != (crop_s, crop_s):
                temp_patch = Image.new('RGB', (crop_s, crop_s), (255, 255, 255))
                temp_patch.paste(patch, (x1 - int(center_x - crop_s // 2), y1 - int(center_y - crop_s // 2)))
                patch = temp_patch
                
            if patch.size != (target_s, target_s):
                patch = patch.resize((target_s, target_s), Image.BILINEAR)
            return patch
        
        total_spots_count = 0
        for wsi_file in wsi_filenames:
            wsi_img = Image.open(wsi_file)

            wsi_basename = os.path.splitext(wsi_file)[0]
            st_feats_path = wsi_basename + '_stdata.tsv'
            try:
                st_spots_pixel_map_path = wsi_basename.split('_P')[0] + '_spot_data-selection-P' + wsi_basename.split('_P')[1] + '.tsv'
            except IndexError:
                continue
            
            if not (os.path.exists(st_feats_path) and os.path.exists(st_spots_pixel_map_path)):
                continue

            feats_all = pandas.read_csv(st_feats_path, sep='\t', index_col=0, header=0)
            spots_pixel_map_all = pandas.read_csv(st_spots_pixel_map_path, sep='\t', header=0)

            current_wsi_spots = []
            valid_spot_indices = []
            
            for i, spot_info in spots_pixel_map_all.iterrows():
                idx = f"{int(spot_info['x'])}x{int(spot_info['y'])}"
                center_x, center_y = spot_info['pixel_x'], spot_info['pixel_y']
                
                try:
                    feats = np.array(feats_all.loc[idx][selected_genes])
                except KeyError:
                    continue
                
                spot_sum = np.sum(feats)
                if spot_sum == 0:
                    continue
                
                feats = np.log1p(feats * 1e6 / spot_sum).astype(np.float32)
                
                current_wsi_spots.append({
                    'id': idx, 
                    'coords': (center_x, center_y), 
                    'feats': feats, 
                    'index_in_wsi': i
                })
                valid_spot_indices.append(i)
            
            if not current_wsi_spots:
                continue

            all_coords = np.array([s['coords'] for s in current_wsi_spots])
            
            for i in range(len(current_wsi_spots)):
                target_spot = current_wsi_spots[i]
                target_coords = target_spot['coords']
                
                distances = np.linalg.norm(all_coords - np.array(target_coords), axis=1)
                num_neighbors_to_find = min(self.num_neighbors, len(current_wsi_spots))
                nearest_indices = np.argsort(distances)[:num_neighbors_to_find]
                
                patches_9 = []
                feats_9 = []
                
                for neighbor_idx_in_current_wsi in nearest_indices:
                    neighbor_spot = current_wsi_spots[neighbor_idx_in_current_wsi]
                    neighbor_coords = neighbor_spot['coords']
                    
                    try:
                        patch = get_patch_and_resize(wsi_img, neighbor_coords[0], neighbor_coords[1], self.patch_size, self.patch_size)
                        patches_9.append(patch)
                        feats_9.append(neighbor_spot['feats'])
                    except Exception as e:
                        if self.verbose: print(f'裁剪邻居图像时出错: {e}, spot: {neighbor_spot["id"]}')
                        patches_9 = []
                        feats_9 = []
                        break
                
                if patches_9:
                    num_found = len(patches_9)
                    if num_found < self.num_neighbors:
                        padding_patch = Image.new('RGB', (self.patch_size, self.patch_size), (255, 255, 255))
                        patches_9.extend([padding_patch] * (9 - num_found))
                        
                        num_genes = len(target_spot['feats'])
                        padding_feats = np.zeros(num_genes, dtype=np.float32)
                        feats_9.extend([padding_feats] * (self.num_neighbors - num_found))

                    spot_entry = {
                        'spot_info': [target_spot['id'], target_coords[0], target_coords[1]],
                        'img_center': patches_9[0],
                        'feats_center': feats_9[0],
                        'hypergraph_x': patches_9,
                        'hypergraph_x_exp': np.array(feats_9)
                    }

                    if self.geneformer_emb_dir is not None:
                        wsi_basename = os.path.splitext(os.path.basename(wsi_file))[0]
                        wsi_gf = self.geneformer_lookup.get(wsi_basename, {})
                        gf_center = wsi_gf.get(target_spot['id'], None)
                        if gf_center is None:
                            gf_center = np.zeros(self.geneformer_dim, dtype=np.float32)
                        gf_neighbors = []
                        for neighbor_idx_in_list in nearest_indices:
                            nb_id = current_wsi_spots[neighbor_idx_in_list]['id']
                            gf_nb = wsi_gf.get(nb_id, None)
                            if gf_nb is None:
                                gf_nb = np.zeros(self.geneformer_dim, dtype=np.float32)
                            gf_neighbors.append(gf_nb)
                        num_found_gf = len(gf_neighbors)
                        if num_found_gf < self.num_neighbors:
                            gf_neighbors.extend(
                                [np.zeros(self.geneformer_dim, dtype=np.float32)] * (self.num_neighbors - num_found_gf)
                            )
                        spot_entry['gf_center'] = gf_center.astype(np.float32)
                        spot_entry['gf_neighbors'] = np.array(gf_neighbors, dtype=np.float32)

                    self.all_spot_data.append(spot_entry)
                    total_spots_count += 1

        print(f'--- [Fold {fold_index+1}/{k_folds}, Mode: {self.mode}] 加载完成，总样本数: {total_spots_count} ---')

    def _load_geneformer_embeddings(self, emb_dir):
        import glob as _glob
        emb_files = sorted(_glob.glob(os.path.join(emb_dir, '*_geneformer_emb.npy')))
        for emb_file in emb_files:
            basename = os.path.basename(emb_file).replace('_geneformer_emb.npy', '')
            ids_file = os.path.join(emb_dir, f"{basename}_geneformer_spotids.json")
            if not os.path.exists(ids_file):
                continue
            embeddings = np.load(emb_file)
            with open(ids_file, 'r') as f:
                spot_ids = json.load(f)
            if self.geneformer_dim == 0:
                self.geneformer_dim = embeddings.shape[1]
            wsi_lookup = {}
            for i, sid in enumerate(spot_ids):
                wsi_lookup[sid] = embeddings[i]
            self.geneformer_lookup[basename] = wsi_lookup
        total = sum(len(v) for v in self.geneformer_lookup.values())
        print(f"--- Geneformer embeddings loaded: {len(self.geneformer_lookup)} WSIs, {total} spots, dim={self.geneformer_dim} ---")

    def __len__(self):
        return len(self.all_spot_data)

    def augmentation(self, img):
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
        if self.mode == 'train':
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
        
        if self.geneformer_emb_dir is not None and 'gf_center' in data:
            gf_center_tensor = torch.from_numpy(data['gf_center']).float()
            gf_neighbors_tensor = torch.from_numpy(data['gf_neighbors']).float()
            return (data['spot_info'],
                    img_center_tensor,
                    feats_center_tensor,
                    hypergraph_x,
                    hypergraph_x_exp,
                    gene_text_encoding_tensor,
                    gf_center_tensor,
                    gf_neighbors_tensor
                    )

        return (data['spot_info'],
                img_center_tensor,
                feats_center_tensor,
                hypergraph_x,
                hypergraph_x_exp,
                gene_text_encoding_tensor
                )
