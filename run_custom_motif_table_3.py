import time
import torch
#print(torch.__version__)
#time.sleep(999)

from dig_motif_table_3.threedgraph.dataset import QM93D
from dig_motif_table_3.threedgraph.dataset import MD17
from dig_motif_table_3.threedgraph.method import Custom_Model
from dig_motif_table_3.threedgraph.method import run
from dig_motif_table_3.threedgraph.evaluation import ThreeDEvaluator


##conda activate QM9_env_foreign_v2

import torch
import numpy as np
import random

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

seed_num = 42
#set_seed(seed_num)

#conda activate QM9_env_foreign

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device

import sys
target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']

print(sys.argv[1])

import random

#7 - U0
#0 - mu
#2 - homo
#11 - Cv
#python run_custom_motif_table_3.py 8 "./chk_files/chk_04_03_2024_10_49_PM" 4 32 128
dataset = QM93D(root='dataset/')
target = target_list[int(sys.argv[1])]#'Cv'#['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']
dataset.data.y = dataset.data[target]

chk_filename = str(sys.argv[2])

var_num_patches = int(sys.argv[3])
var_effective_patch_dim = int(sys.argv[4])
var_total_feature_dim = int(sys.argv[5])

print(dataset)
#time.sleep(999)
print("Y Label analysis")
print(dataset.data.y)
print("torch.max: "+str(torch.max(dataset.data.y)))
print("torch.min: "+str(torch.min(dataset.data.y)))
print("torch.mean: "+str(torch.mean(dataset.data.y)))

#time.sleep(999)

#split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000,seed=seed_num)#seed = 42
#split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=100, valid_size=10000,seed=seed_num)#seed = 42

split_idx = torch.load("./attn_bias_data/split_idx.pt")


print(split_idx)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validation, test:', len(train_dataset), len(valid_dataset), len(test_dataset))
#train_dataset=train_dataset[0:1000]
#valid_dataset=valid_dataset[0:1000]
#test_dataset=test_dataset[0:1000]



#model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4, 
#        hidden_channels=128, out_channels=1, int_emb_size=64, 
#        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
#        num_spherical=3, num_radial=6, envelope_exponent=5, 
#        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
#        )

#model = Custom_Model(var_effective_patch_dim=var_effective_patch_dim, var_num_patches = var_num_patches, var_total_feature_dim=var_total_feature_dim)#DimeNetPP()#


#slot_0 = "e2", slot_1 = "e2"

#model = Custom_Model(slot_0 = str(sys.argv[6]), slot_1 = str(sys.argv[7]))#DimeNetPP()#
model = Custom_Model()#DimeNetPP()#
print(model)

#print("m_373_task_"+target+"_var_num_patches_"+str(var_num_patches)+"_var_effective_patch_dim_"+str(var_effective_patch_dim)+"_var_total_feature_dim_"+str(var_total_feature_dim))
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()


run3d = run()
###Default used by Dimenetpp
#run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=150, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,log_dir='./log_folder',name_run=target,seed_id_for_record = seed_num)



#Code_filename:sweeping-task-r2-lr-5e-05-lr_decay_factor-0.1-lr_decay_step_size-15-batch_size-16run_custom.py 
run3d.run(device, 
          train_dataset, 
          valid_dataset, 
          test_dataset, 
          model, 
          loss_func, 
          evaluation, 
          epochs=150, 
          batch_size=32, #32
          vt_batch_size=16,#32 
          lr=5e-04,#5e-04,#5e-04,#0.0005, #1e-04
          lr_decay_factor=0.5, 
          lr_decay_step_size=15,#15
          weight_decay=0.0,
          log_dir='./log_folder',
          name_run=target,
          seed_id_for_record = seed_num, 
          save_dir=chk_filename,#"./chk_files/chk_06_12_2023_04_26_PM",
          mean_train = torch.mean(train_dataset.data.y), 
          std_train = torch.std(train_dataset.data.y),
          #use_chk_path = "chk_files/chk_03_03_2024_05_24_PM/valid_checkpoint_epoch_29.pt"#None
          #use_chk_path = "chk_files/chk_dime/valid_checkpoint_epoch_dime.pt"#None
          #use_chk_path = "chk_files/chk_dt3/valid_checkpoint_epoch_dt3.pt"#None
          
          )