import os
from glob import glob
import warnings
import pandas as pd
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import scprep as scp

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaselineDataset, self).__init__()
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.features_train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.RandomApply([transforms.RandomRotation((0, 180))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.features_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

class STDataset(BaselineDataset):
    def __init__(self, mode: str, fold: int = 0, extract_mode: str = None, test_data=None, **kwargs):
        super().__init__()
        self.use_pyvips = kwargs['use_pyvips']
        self.r = kwargs['radius'] // 2
        self.mode = mode
        if mode in ["external_test", "inference"]:
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/test/{self.data}"
        elif mode == "extraction":
            self.extract_mode = extract_mode
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/{self.data}"
            self.datatype = f"{kwargs['datatype']}"
        else:
            self.data = kwargs['type']
            if self.data == 'her2st':
                self.data_dir = f"{kwargs['data_dir']}"
            if self.data == 'stnet':
                self.data_dir = f"{kwargs['data_dir']}"
            if self.data == 'skin':
                self.data_dir = f"{kwargs['data_dir']}"
            if self.data == 'pcw':
                self.data_dir = f"{kwargs['data_dir']}"
            if self.data == 'mouse':
                self.data_dir = f"{kwargs['data_dir']}"
        
        if self.data == 'her2st':
            names = ['A','B','C','D','E','F','G','H']
            
        if self.data == 'stnet':
            names = ['E','F','I','J','L','M','N','O','P','R','S','T','U','V','W']
        
        if self.data == 'skin':
            names = ['A','B','C','D']
            
        if self.data == 'pcw':
            names = ['A','B','C','D','E','F']
        
        if self.data == 'mouse':
            names = ['A','B','C','D']
        
        te_names = []
        if self.data == 'her2st':
            patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            te_names = [i for i in names if patients[fold] in i]
        
        if self.data == 'stnet':
            patients = names = ['E','F','I','J','L','M','N','O','P','R','S','T','U','V','W']
            num_folds = 15
            fold_size = len(patients) // num_folds
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < num_folds - 1 else len(patients)
            te_names.extend(patients[start_idx:end_idx])
            if len(te_names) > 2:
                te_names = te_names[:2]
        
        if self.data == 'skin':
            patients = ['A','B','C','D']
            num_folds = 4
            fold_size = len(patients) // num_folds
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < num_folds - 1 else len(patients)
            te_names.extend(patients[start_idx:end_idx])
        
        if self.data == 'pcw':
            patients = ['A','B','C','D','E','F']
            num_folds = 6
            fold_size = len(patients) // num_folds
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < num_folds - 1 else len(patients)
            te_names.extend(patients[start_idx:end_idx])
        
        if self.data == 'mouse':
            patients = ['A','B','C','D']
            num_folds = 4
            fold_size = len(patients) // num_folds
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < num_folds - 1 else len(patients)
            te_names.extend(patients[start_idx:end_idx])
        
        tr_names = [name for name in names if name not in te_names]
        print(f"tests: {te_names}")
        print(f"train: {tr_names}")
        if self.mode == 'train':
            self.names = tr_names
        else:
            self.names = te_names
        
        self.gene_names = []
        if self.data == 'her2st':
            gene_file = self.data_dir + '/her2st_top_250_genes.csv'
            df = pd.read_csv(gene_file)
            self.gene_names = df['Gene'].tolist()
        
        if self.data == 'stnet':
            gene_file = self.data_dir + '/stnet_top_250_genes.csv'
            df = pd.read_csv(gene_file)
            self.gene_names = df['Gene'].tolist()
        
        if self.data == 'skin':
            gene_file = self.data_dir + '/skin_top_250_genes.csv'
            df = pd.read_csv(gene_file)
            self.gene_names = df['Gene'].tolist()
        
        if self.data == 'pcw':
            gene_file = self.data_dir + '/pcw_top_250_genes.csv'
            df = pd.read_csv(gene_file)
            self.gene_names = df['Gene'].tolist()
        
        if self.data == 'mouse':
            gene_file = self.data_dir + '/mouse_top_250_genes.csv'
            df = pd.read_csv(gene_file)
            self.gene_names = df['Gene'].tolist()
        self.graphs = []
        for name in self.names:
            
            file = kwargs['data_dir'] + '/' + name + '_all_layer_data.npy'
            try:
                data = np.load(file, allow_pickle=True).item()
                for row_key, row_value in data.items():
                    center_spot_img=None 
                    center_expression=None 
                    all_spots_imgs=[]
                    all_expressions=[]
                    for layer_key, layer_value in row_value.items():
                        gene_expressions = layer_value.get("gene_expressions", [])
                        cropped_image_names = layer_value.get("cropped_image_names", [])
                        if not gene_expressions or not cropped_image_names:
                            continue
                        filtered_gene_expressions = []
                        filtered_cropped_images = []
                        for expr, img_name in zip(gene_expressions, cropped_image_names):
                            if len(expr) == 0:
                                continue    
                            filtered_gene_expressions.append(expr)
                            filtered_cropped_images.append(img_name)
                        if not center_spot_img and not center_expression and filtered_cropped_images:
                            try:
                                center_spot_img = filtered_cropped_images[0]
                                center_expression = {gene: filtered_gene_expressions[0][i] for i, gene in enumerate(self.gene_names)}
                            except IndexError as e:
                                continue
                        all_spots_imgs.extend(filtered_cropped_images)
                        for expr in filtered_gene_expressions:
                            if len(expr) < len(self.gene_names):
                                continue
                            all_expressions.append({gene: expr[i] for i, gene in enumerate(self.gene_names)})
                    if center_spot_img and center_expression:
                        graph = {
                            "center": {"img": center_spot_img, "expression": center_expression},
                            "all": [{"img": img, "expression": expr} for img, expr in zip(all_spots_imgs, all_expressions)]
                        }
                    self.graphs.append(graph)    
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
    def __getitem__(self, index):
        if self.data == 'her2st':
            self.img_dir = self.data_dir + '/cropped_imgs'
        if self.data == 'stnet':
            self.img_dir = self.data_dir + '/cropped_imgs'
        if self.data == 'skin':
            self.img_dir = self.data_dir + '/cropped_imgs'
        if self.data == 'pcw':
            self.img_dir = self.data_dir + '/cropped_imgs'
        if self.data == 'mouse':
            self.img_dir = self.data_dir + '/cropped_imgs'
        graph = self.graphs[index]
        center_img_path = os.path.join(self.img_dir, graph['center']['img'])
        center_img_path = center_img_path.replace('.jpg','.png')
        center_expression = graph['center']['expression']
        center_img = cv2.imread(center_img_path)
        if center_img is not None:
            center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB) 
            if self.mode=='train':
               center_img=self.train_transforms(center_img)
            else:
               center_img=self.test_transforms(center_img)
        all_imgs = []
        all_expressions = []
        
        for spot in graph['all']:
            img_path = os.path.join(self.img_dir, spot['img'])
            img_path = img_path.replace('.jpg','.png')
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.mode=='train':
                   img=self.train_transforms(img)
                else:
                   img=self.test_transforms(img)
            if img is not None and spot.get('expression'):
                all_imgs.append(img)
                all_expressions.append(spot['expression'])
        patches = []
        patches.append(center_img)
        patches = torch.stack(patches, dim=0)
        exps = torch.tensor(list(center_expression.values()),dtype=torch.float32)
        node_features_list = []
        exp_features_list = []
        for idx in all_imgs:
            node_features_list.append(idx)
        for expression in all_expressions:
            exp_tensor = torch.tensor(list(expression.values()))
            exp_features_list.append(exp_tensor)
        hypergraph_x = torch.stack(node_features_list, dim=0).squeeze()
        hypergraph_x_exp = torch.stack(exp_features_list, dim=0).squeeze()
             
        return patches,exps,hypergraph_x,hypergraph_x_exp,graph['center']['img']

    def __len__(self):
        return len(self.graphs)

def collate_fn(batch: tuple):
    patch = torch.stack([item[0] for item in batch])
    exp = torch.stack([item[1] for item in batch])
    image_names=[item[4] for item in batch]
    max_length = max(item[2].size(0) for item in batch)

    neighbors_padded = []
    neighbors_exp_padded = []
    pad_sizes = []  

    for item in batch:
        neighbor = item[2]
        neighbor_exp = item[3]
        pad_size_neighbor = max_length - neighbor.size(0)
        pad_size_exp = max_length - neighbor_exp.size(0)
        pad_sizes.append((neighbor.size(0), neighbor_exp.size(0)))
        padded_neighbor = torch.cat([neighbor, torch.zeros(pad_size_neighbor, *neighbor.size()[1:])], dim=0)
        padded_neighbor_exp = torch.cat([neighbor_exp, torch.zeros(pad_size_exp, *neighbor_exp.size()[1:])], dim=0)
        neighbors_padded.append(padded_neighbor)
        neighbors_exp_padded.append(padded_neighbor_exp)
    neighbors_stacked = torch.stack(neighbors_padded)
    neighbors_exp_stacked = torch.stack(neighbors_exp_padded)
    return patch, exp, neighbors_stacked, neighbors_exp_stacked, pad_sizes,image_names
