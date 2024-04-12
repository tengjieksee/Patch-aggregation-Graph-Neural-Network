import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DC_measure(nn.Module):#https://github.com/zhenxingjian/Partial_Distance_Correlation/blob/main/Partial_Distance_Correlation.ipynb
    def __init__(self):
        super(DC_measure, self).__init__()


    def Distance_Correlation(self, latent, control):#latent = (128, num_atoms), control = (128, num_atoms)
        num_atoms = latent.shape[1]
        #print(latent.shape)#10,1
        #print(control.shape)#10,1
        #print(latent.unsqueeze(0).shape)#1,10,1
        #print(latent.unsqueeze(1).shape)#10,1,1
        #print(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1).shape)#10,10
        #print(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1))#10,10
        #print(torch.square(latent.unsqueeze(0).expand(10,-1,-1) - latent.unsqueeze(1).expand(-1,10,-1)).shape)#10,10,1
        #print(torch.square(latent.unsqueeze(0).expand(10,-1,-1) - latent.unsqueeze(1).expand(-1,10,-1)))#10,10,1
        matrix_a = torch.sqrt(torch.square(latent.unsqueeze(0).expand(10,-1,-1) - latent.unsqueeze(1).expand(10,-1,-1)) + 1e-12)
        matrix_b = torch.sqrt(torch.square(control.unsqueeze(0).expand(10,-1,-1) - control.unsqueeze(1).expand(10,-1,-1)) + 1e-12)
        
        #matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
        #matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

        #print(matrix_a.shape)#10,10
        #print(matrix_b.shape)#10,10


        matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a.reshape(-1,num_atoms))
        matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b.reshape(-1,num_atoms))
        #print("aft")
        #print(matrix_A.shape)#10,10
        #print(matrix_B.shape)#10,10

        XY_mat = matrix_A * matrix_B
        XX_mat = matrix_A * matrix_A
        YY_mat = matrix_B * matrix_B

        Gamma_XY = torch.sum(XY_mat.reshape(-1,num_atoms),0)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(XX_mat.reshape(-1,num_atoms),0)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(YY_mat.reshape(-1,num_atoms),0)/ (matrix_A.shape[0] * matrix_A.shape[1])
        #print(Gamma_XY.shape)#1,1
        #print(Gamma_XX.shape)#1,1
        #print(Gamma_YY.shape)#1,1

        correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r


    def forward(self, latent, control):
        dc_loss = self.Distance_Correlation(latent, control)

        return dc_loss

Distance_Correlation = DC_measure()
#num_atoms,128 to num_atoms,num_atoms
batch_size = 10
x = np.linspace(-3, 3, num=batch_size)#10,1
y = np.random.randn(batch_size)#10,1
y = y + x**2


x = torch.Tensor(x)
x = x.reshape([batch_size,-1])
y = torch.Tensor(y)
y = y.reshape([batch_size,-1])


x = torch.zeros(10,8)-1#dim, num_atoms
y = torch.zeros(10,8)+1
dc = Distance_Correlation(x,y)
print(dc)