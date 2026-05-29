---

<div align="center">
  <img src="https://github.com/tengjieksee/Patch-aggregation-Graph-Neural-Network/assets/47586439/2fc4eac2-88dc-4852-bc24-86b0c7bb12e6" alt="Patch Aggregation for Graph Neural Network">
</div>

# Graph Neural Network based Molecular Property Prediction with Patch Aggregation

This repository hosts the implementation of Patch Aggregation for Graph Neural Network, developed for chemical property prediction.

### Paper

The paper can be found in [here](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00798).

### Abstract

Graph Neural Networks (GNNs) have emerged as powerful tools for quantum chemical property prediction, leveraging the inherent graph structure of molecular systems. GNNs depend on edge-to-node aggregation mechanism for combining edge representations into node representations. Unfortunately, existing learnable edge-to-node aggregation methods substantially increase the number of parameters and thus the computational cost relative to simple sum aggregation. Worse, as we report here, they often fail to improve the predictive accuracy. We therefore propose a novel learnable edge-to-node aggregation mechanism that aims to improve the accuracy and parameter efficiency of GNNPs in predicting molecular properties. The new mechanism, called “patch aggregation”, is inspired by the multi-head attention and mixture-of-experts machine learning techniques. We have incorporated the patch aggregation method into the specialized, state-of-the-art GNN models SchNet, DimeNet++, and SphereNet and show that patch aggregation consistently outperforms existing learnable aggregation techniques (multi-layer perceptron, softmax and set transformer aggregation) in the prediction of molecular properties such as QM9 thermodynamic properties and MD17 molecular dynamics energy and force trajectories. We also find that patch aggregation not only improves prediction accuracy but also enhanced parameter efficiency, making it an attractive option for practical applications where computational resources are limited. Further, we show that Patch aggregation improves accuracy across different GNNP models. Overall, Patch aggregation is a powerful edge-to-node aggregation mechanism that improves the accuracy of molecular property predictions by GNNPs.

## Authors

- **Teng Jiek See**
  - Medicinal Chemistry, Monash Institute of Pharmaceutical Sciences, Monash University, Australia.
- **Daokun Zhang**
  - School of Computer Science, University of Nottingham Ningbo China, China.
- **Mario Boley**
  - Department of Data Science and AI, Faculty of Information Technology, Monash University, Australia.
- **David Chalmers**
  - Medicinal Chemistry, Monash Institute of Pharmaceutical Sciences, Monash University, Australia.


## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- pip package manager

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/tengjieksee/Patch-aggregation-Graph-Neural-Network.git
cd Patch-aggregation-Graph-Neural-Network

# 2. (Recommended) Create a conda environment
conda create -n patchgnn python=3.10 -y
conda activate patchgnn

# 3. Install PyTorch with CUDA support (adjust cuda version as needed)
# For CUDA 11.7:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# 4. Install project dependencies
pip install -r requirements.txt
```

> 💡 **Note**: The `requirements.txt` includes specific versions of PyTorch Geometric extensions (`torch-scatter`, `torch-sparse`, etc.) compiled for CUDA 11.7. If you use a different CUDA version, you may need to reinstall these packages from the [PyG wheels index](https://data.pyg.org/whl/).

---

## 🚀 Quick Start

### TL;DR - Run in 3 Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run QM9 experiment (target: HOMO energy, index=2)
cd qm9_test
python run_custom_motif_table_3.py 2 "./chk_files/qm9_homo_exp" 4 32 128

# 3. Run MD17 experiment (aspirin dataset)
cd ../md17_test
python run_custom_MD17.py 0 "./chk_files/md17_aspirin_exp"
```

---

## 📖 Usage Guide

### QM9 Dataset Experiments

The QM9 dataset contains ~130,000 small organic molecules with 12 regression targets [[3]].

#### Command Format
```bash
cd qm9_test
python run_custom_motif_table_3.py <TARGET_IDX> <CHECKPOINT_DIR> <NUM_PATCHES> <PATCH_DIM> <FEATURE_DIM>
```

#### Arguments Explained
| Argument | Description | Example Values |
|----------|-------------|---------------|
| `TARGET_IDX` | Index of target property (0-11) | `0`=mu, `2`=homo, `11`=Cv |
| `CHECKPOINT_DIR` | Directory to save model checkpoints | `"./chk_files/exp_001"` |
| `NUM_PATCHES` | Number of patches for aggregation | `4`, `8`, `16` |
| `PATCH_DIM` | Dimension per patch | `32`, `64` |
| `FEATURE_DIM` | Total edge feature dimension | `128`, `256` |

#### Target Properties (QM9)
```python
target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
# Indices:      0      1        2       3       4      5      6      7    8    9   10   11
```

#### Example: Train on HOMO Energy
```bash
cd qm9_test
# Run with: target=homo(2), 4 patches, 32-dim patches, 128-dim features
python run_custom_motif_table_3.py 2 "./chk_files/qm9_homo_patch4" 4 32 128
```

#### Training Configuration (Default)
- **Epochs**: 150
- **Batch size**: 32 (training), 16 (validation)
- **Learning rate**: 5e-4 with decay (factor=0.5, step=15)
- **Optimizer**: Adam (weight_decay=0.0)
- **Loss**: L1Loss (MAE)

---

### MD17 Dataset Experiments

The MD17 dataset contains molecular dynamics trajectories for energy and force prediction [[25]].

#### Command Format
```bash
cd md17_test
python run_custom_MD17.py <DATASET_IDX> <CHECKPOINT_DIR>
```

#### Dataset Options
| Index | Molecule | Dataset Name |
|-------|----------|-------------|
| 0 | Aspirin | `md17_aspirin` |
| 1 | Benzene (2017) | `md17_benzene2017` |
| 2 | Ethanol | `md17_ethanol` |
| 3 | Malonaldehyde | `md17_malonaldehyde` |
| 4 | Naphthalene | `md17_naphthalene` |
| 5 | Salicylic Acid | `md17_salicylic` |
| 6 | Toluene | `md17_toluene` |
| 7 | Uracil | `md17_uracil` |

#### Example: Train on Aspirin
```bash
cd md17_test
# Run with: dataset=aspirin(0), custom checkpoint path
python run_custom_MD17.py 0 "./chk_files/md17_aspirin_patch"
```

#### Training Configuration (Default)
- **Epochs**: 2000
- **Batch size**: 8 (training), 64 (validation)
- **Learning rate**: 5e-4 with decay (factor=0.5, step=200)
- **Tasks**: Energy AND force prediction (`energy_and_force=True`)
- **Loss**: L1Loss (MAE) for both energy and forces

---

## 💻 Patch Aggregation Core Code

The core patch aggregation mechanism can be integrated into any GNN layer:

```python
import torch
from torch_geometric.utils import scatter

def patch_aggregation(edge_tensor, index, 
                      num_patches=4, 
                      patch_dim=32, 
                      feature_dim=128):
    """
    Apply patch aggregation to edge features.
    
    Args:
        edge_tensor: Tensor of shape [num_edges, feature_dim]
        index: Node indices for scatter operation [num_edges]
        num_patches: Number of patches to split features into
        patch_dim: Dimension of each patch
        feature_dim: Total feature dimension (must = num_patches * patch_dim)
    
    Returns:
        Aggregated node features of shape [num_nodes, feature_dim]
    """
    # Initialize learnable projection for patch weights
    fc_all = torch.nn.Linear(patch_dim, feature_dim, bias=False)
    
    # Reshape edges into patches: [E, P, D_patch]
    e_patches = edge_tensor.reshape(-1, num_patches, patch_dim)
    
    # Generate patch attention weights (clamped to [0,1])
    weights = torch.clamp(fc_all(e_patches), 0., 1.)  # [E, P, D_feature]
    
    # Split weights back into per-patch tensors
    weight_list = [
        weights[:, :, i * patch_dim : (i + 1) * patch_dim] 
        for i in range(num_patches)
    ]
    
    # Apply weighted sum across patches
    output_list = [torch.zeros(edge_tensor.shape[0], 1)]  # placeholder
    for i, w in enumerate(weight_list):
        weighted = e_patches * w  # [E, P, D_patch]
        summed = weighted.sum(1)   # [E, D_patch]
        output_list.append(summed)
    
    # Concatenate and remove placeholder
    updated_edges = torch.cat(output_list, dim=-1)[:, 1:]  # [E, feature_dim]
    
    # Aggregate to nodes using scatter
    node_features = scatter(updated_edges, index, dim=0, reduce='sum')
    
    return node_features
```

### Key Design Choices
1. **Patch Splitting**: Edge features are reshaped into `num_patches × patch_dim` chunks (inspired by multi-head attention)
2. **Learnable Weights**: A linear layer generates attention-like weights for each patch
3. **Clamping**: Weights are clamped to [0,1] for stable training
4. **Weighted Aggregation**: Each patch is weighted and summed, then concatenated
5. **Node Aggregation**: Final edge features are scattered to nodes using PyG's `scatter`

---

## 🏗️ Model Architecture

The repository supports multiple GNN backbones with patch aggregation:

```
qm9_test/
├── dig_motif_table_3/          # QM9-specific modules
│   └── threedgraph/
│       ├── dataset.py          # QM93D dataset loader
│       ├── method.py           # Custom_Model with patch aggregation
│       ├── run.py              # Training loop
│       └── evaluation.py       # MAE/RMSE metrics
├── run_custom_motif_table_3.py # Main QM9 experiment script
└── tensornet/                  # TensorNet model variants

md17_test/
├── dig_MD17/                   # MD17-specific modules
│   └── threedgraph/
│       ├── dataset.py          # MD17 dataset loader
│       ├── method.py           # SchNet/DimeNetPP/SphereNet + patch agg
│       ├── run.py              # Training loop with force prediction
│       └── evaluation.py       # Energy/force evaluation metrics
├── run_custom_MD17.py          # Main MD17 experiment script
└── ...
```

### Supported Base Models
- **SchNet**: Continuous-filter convolutional GNN
- **DimeNet++**: Directional message passing with angular features
- **SphereNet**: Spherical message passing with torsion angles
- **TensorNet**: Tensor-based equivariant GNN
- **ViSNet**: Visual-inspired geometric GNN

---

## 📊 Expected Results

When properly configured, patch aggregation should show:
- **QM9**: 5-15% MAE reduction compared to sum aggregation across targets
- **MD17**: Improved energy/force prediction with fewer parameters
- **Efficiency**: Comparable or better accuracy with reduced parameter count vs. MLP/softmax aggregation

> 📈 Results may vary based on hyperparameters, random seeds, and hardware. Use the provided `seed_num` for reproducibility.

---

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch sizes in run().run() call:
batch_size=16, vt_batch_size=8  # Instead of 32/16
```

**PyG Extension Version Mismatch**
```bash
# Reinstall PyG extensions for your CUDA version:
pip uninstall torch-scatter torch-sparse torch-cluster
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

**Dataset Download Fails**
```bash
# Manually download datasets to dataset/ folder:
# QM9: https://figshare.com/articles/dataset/QM9/1308312
# MD17: http://quantum-machine.org/gdml/data/npz/
```

**Deterministic Training**
```python
# Ensure reproducibility (already in scripts):
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{see2024patchaggregation,
  title={Graph Neural Network-Based Molecular Property Prediction with Patch Aggregation},
  author={See, Teng Jiek and Zhang, Daokun and Boley, Mario and Chalmers, David},
  journal={Journal of Chemical Theory and Computation},
  year={2024},
  publisher={American Chemical Society},
  doi={10.1021/acs.jctc.4c00798}
}
```


## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

> ⚠️ **Disclaimer**: This code is provided for research purposes. The authors make no warranties regarding fitness for any particular purpose.


