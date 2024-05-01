import time
import torch
from dig_motif_table_3.threedgraph.dataset import QM93D
from dig_motif_table_3.threedgraph.dataset import MD17
from dig_motif_table_3.threedgraph.method import Custom_Model
from dig_motif_table_3.threedgraph.method import run
from dig_motif_table_3.threedgraph.evaluation import ThreeDEvaluator
import torch
import numpy as np
import random
import os
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed_num = 42
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device
target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']

dataset = QM93D(root='dataset/')
target = target_list[int(sys.argv[1])]#'Cv'#['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']
dataset.data.y = dataset.data[target]

chk_filename = str(sys.argv[2])

var_num_patches = int(sys.argv[3])
var_effective_patch_dim = int(sys.argv[4])
var_total_feature_dim = int(sys.argv[5])

split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000,seed=seed_num)#seed = 42

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validation, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

model = Custom_Model()

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()


run3d = run()

run3d.run(device, 
          train_dataset, 
          valid_dataset, 
          test_dataset, 
          model, 
          loss_func, 
          evaluation, 
          epochs=150, 
          batch_size=32,
          vt_batch_size=16,
          lr=5e-04,
          lr_decay_factor=0.5, 
          lr_decay_step_size=15,
          weight_decay=0.0,
          log_dir='./log_folder',
          name_run=target,
          seed_id_for_record = seed_num, 
          save_dir=chk_filename,#e.g. "./chk_files/chk_06_12_2023_04_26_PM",
          mean_train = torch.mean(train_dataset.data.y), 
          std_train = torch.std(train_dataset.data.y),
          use_chk_path = None,#e.g. "chk_files/chk_03_03_2024_05_24_PM/valid_checkpoint_epoch_29.pt"
          )