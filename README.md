---

<div align="center">
  <img src="https://github.com/tengjieksee/Patch-aggregation-Graph-Neural-Network/assets/47586439/2fc4eac2-88dc-4852-bc24-86b0c7bb12e6" alt="Patch Aggregation for Graph Neural Network">
</div>

# Graph Neural Network based Chemical Property Prediction with Patch Aggregation

This repository hosts the implementation of Patch Aggregation for Graph Neural Network, developed for chemical property prediction.

### Paper

The paper can be found in xxx.

### Abstract

Graph Neural Network Potentials (GNNPs) have emerged as powerful tools for quantum chemical property prediction, leveraging the inherent graph structure of molecular systems. GNNPs depend on edge-to-node aggregation mechanism for combining edge representations into node representations. Unfortunately, existing learnable edge-to-node aggregation methods substantially increase the number of parameters and thus the computational cost relative to simple sum aggregation. Worse, as we report here, they often fail to improve the predictive accuracy. We therefore propose a novel learnable edge-to-node aggregation mechanism that aims to improve the accuracy and parameter efficiency of GNNPs in predicting molecular properties. The new mechanism, called “patch aggregation”, is inspired by the multi-head attention and mixture-of-experts machine learning techniques. We have incorporated the patch aggregation method into the specialized, state-of-the-art GNNP models SchNet, DimeNet++, and SphereNet and show that patch aggregation consistently outperforms existing learnable aggregation techniques (multi-layer perceptron, softmax and set transformer aggregation) in the prediction of molecular properties such as QM9 thermodynamic properties and MD17 molecular dynamics energy and force trajectories. We also find that patch aggregation not only improves prediction accuracy but also enhanced parameter efficiency, making it an attractive option for practical applications where computational resources are limited. Further, we show that Patch aggregation improves accuracy across different GNNP models. Overall, Patch aggregation is a powerful edge-to-node aggregation mechanism that improves the accuracy of molecular property predictions by GNNPs.

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





