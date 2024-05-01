import time
import torch
#print(torch.__version__)
#time.sleep(999)
import sys
from dig_MD17.threedgraph.dataset import QM93D
from dig_MD17.threedgraph.dataset import MD17
from dig_MD17.threedgraph.method import SchNet, SphereNet, DimeNetPP, ComENet, Custom_Model
from dig_MD17.threedgraph.method import run
from dig_MD17.threedgraph.evaluation import ThreeDEvaluator


##conda activate QM9_env_foreign_v2

import torch
import numpy as np
import random

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

seed_num = 960#int(sys.argv[3])
#set_seed(seed_num)

#conda activate QM9_env_foreign

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device

if int(sys.argv[1]) == 0:
    name_dataset = 'md17_aspirin'
elif int(sys.argv[1]) == 1:
    name_dataset = 'md17_benzene2017'
elif int(sys.argv[1]) == 2:
    name_dataset = 'md17_ethanol'
elif int(sys.argv[1]) == 3:
    name_dataset = 'md17_malonaldehyde'
elif int(sys.argv[1]) == 4:
    name_dataset = 'md17_naphthalene'
elif int(sys.argv[1]) == 5:
    name_dataset = 'md17_salicylic'
elif int(sys.argv[1]) == 6:
    name_dataset = 'md17_toluene'
elif int(sys.argv[1]) == 7:
    name_dataset = 'md17_uracil'

#python run_custom_motif_table_3.py 0 "./chk_files/chk_xxx"
dataset = MD17(root='dataset/', name=name_dataset)

print("Start")
print(name_dataset)

print("m_465_dimenetpp (100 motifs)")
##################http://quantum-machine.org/gdml/data/npz/
#dataset = MD17(root='dataset/', name='md17_aspirin')
#print(len(dataset.data.y))#211762
#dataset = MD17(root='dataset/', name='md17_benzene2017')
#print(len(dataset.data.y))#627983
#dataset = MD17(root='dataset/', name='md17_ethanol')
#print(len(dataset.data.y))#555092
#dataset = MD17(root='dataset/', name='md17_malonaldehyde')
#print(len(dataset.data.y))#993237
#dataset = MD17(root='dataset/', name='md17_naphthalene')
#print(len(dataset.data.y))#326250
#dataset = MD17(root='dataset/', name='md17_salicylic')
#print(len(dataset.data.y))#320231
#dataset = MD17(root='dataset/', name='md17_toluene')
#print(len(dataset.data.y))#442790
#dataset = MD17(root='dataset/', name='md17_uracil')
#print(len(dataset.data.y))#133770

#time.sleep(999)
#[   ]	md17_aspirin.npz	2018-09-10 16:22	193M	 
#[   ]	md17_benzene2017.npz	2018-09-14 11:23	209M	 
#[   ]	md17_ethanol.npz	2018-09-10 16:22	219M	 
#[   ]	md17_malonaldehyde.npz	2018-09-10 16:23	393M	 
#[   ]	md17_naphthalene.npz	2018-09-10 16:23	254M	 
#[   ]	md17_salicylic.npz	2018-09-10 16:23	223M	 
#[   ]	md17_toluene.npz	2018-09-10 16:24	289M	 
#[   ]	md17_uracil.npz	



split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=seed_num)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))


#time.sleep(999)
model = Custom_Model(energy_and_force=True)#DimeNetPP()#
print(model)
print("MD17")
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()


run3d = run()


#Code_filename:sweeping-task-r2-lr-5e-05-lr_decay_factor-0.1-lr_decay_step_size-15-batch_size-16run_custom.py 
run3d.run(device, 
          train_dataset, 
          valid_dataset, 
          test_dataset, 
          model, 
          loss_func, 
          evaluation, 
          epochs=2000, #1000
          batch_size=8,#32, #32
          vt_batch_size=64,#256,#32 
          lr=5e-04,#0.0005, #1e-04
          lr_decay_factor=0.5, 
          lr_decay_step_size=200,#15
          weight_decay=0.0,
          log_dir='./log_folder',
          name_run="md17",
          seed_id_for_record = seed_num, 
          save_dir=str(sys.argv[2]),#"./chk_files/_MD17_chk_03_01_2024_01_37_PM",
          mean_train = torch.mean(train_dataset.data.y), 
          std_train = torch.std(train_dataset.data.y),
          energy_and_force=True,
          #use_chk_path = "chk_files/chk_03_09_2023_01_25_PM/valid_checkpoint_epoch_47.pt"#None
          #use_chk_path = "chk_files/chk_dime/valid_checkpoint_epoch_dime.pt"#None
          #use_chk_path = "chk_files/chk_dt3/valid_checkpoint_epoch_dt3.pt"#None
          
          )