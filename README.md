---

<div align="center">
  <img src="https://github.com/tengjieksee/Patch-aggregation-Graph-Neural-Network/assets/47586439/2fc4eac2-88dc-4852-bc24-86b0c7bb12e6" alt="Patch Aggregation for Graph Neural Network">
</div>

# Patch Aggregation for Graph Neural Network

This repository hosts the implementation of Patch Aggregation for Graph Neural Network, developed for chemical property prediction.

### Paper

The paper can be found in xxx.

## Authors

- **Teng Jiek See**
  - Medicinal Chemistry, Monash Institute of Pharmaceutical Sciences, Monash University, Australia.
- **Daokun Zhang**
  - School of Computer Science, University of Nottingham Ningbo China, China.
- **Mario Boley**
  - Department of Data Science and AI, Faculty of Information Technology, Monash University, Australia.
- **David Chalmers**
  - Medicinal Chemistry, Monash Institute of Pharmaceutical Sciences, Monash University, Australia.

## Usage
- pip install -r `requirements.txt`
- For QM9 test, navigate to the `qm9_test` folder.
- For MD17 test, navigate to the `md17_test` folder.

### TL;DR

The main code for patch aggregation can be found below.

```python
import torch

###Settings
var_num_patches = 4
var_effective_patch_dim = 32
fc_all = torch.nn.Linear(var_effective_patch_dim,128,bias=False)
###End of Settings

#We define our edge tensor. In this case, 500 edges with 128 feature dimensions in each.
edge_tensor = torch.rand(500,128)

#We create the patches. This reshaping ops is inspired by MHA.
e1_0 = edge_tensor.reshape(-1, var_num_patches, var_effective_patch_dim)#(-1,4,32)

#Expand the feature dim (from 32 to 128) via f_all neural network and Restricting the numerical values of the patches to 0-1.
e1_1 = torch.clamp(fc_all(e1_0),0.,1.)#-1,4,4*32 = -1,4,128


e1_lst = [e1_1[:, :, i * var_effective_patch_dim : (i + 1) * var_effective_patch_dim] for i in range(var_num_patches)]

out_lst = torch.zeros(e1_0.shape[0],1)
for _i in range(len(e1_lst)):
    out_ = e1_0 * e1_lst[_i]#(-1,4,32)
    out_ = out_.sum(1) #-1,32
    out_lst = torch.cat((out_lst,out_),-1)
    

#We get the final updated edge tensor
updated_edge_tensor = out_lst[:,1:]#Remove the placeholder by slicing the tensor#5
print(updated_edge_tensor.shape)#500,128

#Essentially summing up all edges with respect to the main node. Need torch.scatter for this.
v = scatter(updated_edge_tensor, index, dim=0)





