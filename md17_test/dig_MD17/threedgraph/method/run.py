from __future__ import division
from __future__ import unicode_literals

from typing import Iterable, Optional
import weakref
import copy
import contextlib

import torch
import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import glob

#import wandb
#wandb.init(project="Project_QCT2")

def delete_past_chk(directory_path):
    #directory_path = "/path/to/your/directory"  # Replace this with the actual path to your directory
    for filename in os.listdir(directory_path):
        if filename.startswith("valid_checkpoint_epoch_"):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {filename}")
            else:
                print(f"Not a file: {filename}")



# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        use_num_updates: bool = True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach()
            for p in parameters
        ]
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )

#from fairscale.experimental.nn.offload import OffloadModel
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR


class LinearWarmupExponentialDecay():
    """This schedule combines a linear warmup with an exponential decay."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, decay_steps: int, decay_rate: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))
        self.decay_scheduler = ExponentialLR(optimizer, gamma=decay_rate**(1/decay_steps))

    def step(self):
        self.warmup_scheduler.step()
        self.decay_scheduler.step()

    def get_lr(self):
        return [base_lr * warmup_factor * decay_factor
                for base_lr, warmup_factor, decay_factor in zip(self.optimizer.param_groups[0]['initial_lr'],
                                                                 self.warmup_scheduler.get_lr(),
                                                                 self.decay_scheduler.get_lr())]


import os
def find_pt_file(save_dir):
    #save_dir = "/path/to/your/directory"  # Replace this with your actual directory path
    
    # Get a list of all files in the directory
    files = os.listdir(save_dir)
    
    # Filter the list to include only files with a .pt extension
    pt_files = [file for file in files if file.endswith(".pt")]
    
    # Print the names of the .pt files
    for pt_file in pt_files:
        print(pt_file)
    return pt_file




import csv

class CustomLogger:
    def __init__(self, csv_filename, header):
        self.csv_filename = csv_filename
        self.header = header
        self.csv_file = None
        self.writer = None

        # Create and write header to the CSV file
        with open(self.csv_filename, mode='w', newline='') as file:
            self.csv_file = file
            self.writer = csv.writer(file)
            self.writer.writerow(self.header)

    def log(self, values):
        # Append values to the CSV file
        with open(self.csv_filename, mode='a', newline='') as file:
            self.writer = csv.writer(file)
            self.writer.writerow(values)


# Example usage
#learning_rate = 0.1
#warmup_steps = 1000
#decay_steps = 10000
#decay_rate = 0.9

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#lr_scheduler = LinearWarmupExponentialDecay(optimizer, warmup_steps, decay_steps, decay_rate)

#for epoch in range(total_epochs):
#    # Train your model
#    lr_scheduler.step()
#    current_lr = lr_scheduler.get_lr()[0]
#    print(f"Epoch [{epoch+1}/{total_epochs}] - Learning Rate: {current_lr:.6f}")
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
    
    def scale_the_metrics_(self,mae,run_name):
        if run_name == "mu" or run_name == "Cv":
            mae = mae
        elif run_name == "alpha":
            mae = mae #bohr^3 which is a_0^3
        elif run_name == "homo" or run_name == "lumo" or run_name == "gap" or run_name == "zpve" or run_name == "U0" or run_name == "U" or run_name == "H" or run_name == "G":
            mae = mae*1000 #eV to milli eV
        elif run_name == "r2":
            mae = mae #bohr^2 which is a_0^2
        
        if run_name == "mu":
            unit_ = "Debye"
        elif run_name == "alpha":
            unit_ = "angstrom^3"
        elif run_name == "homo" or run_name == "lumo" or run_name == "gap" or run_name == "zpve" or run_name == "U0" or run_name == "U" or run_name == "H" or run_name == "G":
            unit_ = "milli eV"
        elif run_name == "r2":
            unit_ = "angstrom^2"
        elif run_name == "Cv":
            unit_ = "cal/(mol K)"
        
        unit_ = "milli eV"
        
        return (mae,unit_)
    
    def lr_lambda_sch_func(step):#scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_sch_func)
        batch_size = 32
        #total_steps = int(110000/batch_size)
        total_training_sample_size = 1000#110000
        total_steps = int(total_training_sample_size/batch_size)
        warmup_steps = 0.1*total_steps
        max_lr = 5e-04
        if step < warmup_steps:
            # Linear warm-up
            return (float(step) / float(max(1, warmup_steps)))*max_lr
        else:
            # Linear decay
            return (1.0 - (float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))))*max_lr
    
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir='', name_run='', seed_id_for_record=42, mean_train = 1000.,std_train = 10.,use_chk_path=None):
        r"""
        The run script for training and validation.
        
        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)
        
        """
        
        model = model.to(device)
        #model.apply(init_weights)
        #model = OffloadModel(
        #    model=model_original,
        #    device=torch.device("cuda"),
        #    offload_device=torch.device("cpu"),
        #    num_slices=3,
        #    #checkpoint_activation=True,
        #    num_microbatches=1,
        #    )
        
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        print({"#Params": num_params})
        print({"#seed": seed_id_for_record})
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
        
        
        num_training_dataset_samples = 1000
        num_epochs = epochs#150
        #batch_size = 32  # Example batch size (adjust as needed)
        warmup_fraction = 0.10  # 10% warmup fraction
        # Calculate total training steps
        total_training_steps = (num_training_dataset_samples / batch_size) * num_epochs
        #print(total_training_steps)
        #time.sleep(999)
        # Calculate warmup steps
        warmup_steps = 1000#total_training_steps * warmup_fraction
        # Calculate decay steps
        #decay_steps = total_training_steps - warmup_steps
        
        # Choose a decay rate (e.g., 0.97 for a 3% decay per step, adjust as needed)
        #decay_rate = 0.97
        
        
        #warmup_steps, decay_steps, decay_rate = 3000, 4000000, 0.01
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
        
        
        #num_training_dataset_samples = 110000
        #num_epochs = 150
        #batch_size = 32  # Example batch size (adjust as needed)
        #warmup_fraction = 0.10  # 10% warmup fraction
        # Calculate total training steps
        #total_training_steps = (num_training_dataset_samples / batch_size) * num_epochs
        #print(total_training_steps)
        #time.sleep(999)
        # Calculate warmup steps
        #warmup_steps = total_training_steps * warmup_fraction
        # Calculate decay steps
        #decay_steps = total_training_steps - warmup_steps
        
        # Choose a decay rate (e.g., 0.97 for a 3% decay per step, adjust as needed)
        #decay_rate = 0.97
        
        
        #warmup_steps, decay_steps, decay_rate = 3000, 4000000, 0.01
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
        
        if use_chk_path != None:
            PATH = use_chk_path#"./chk_files/chk_18_05_2023_12_53_PM/valid_checkpoint_epoch_10.pt"
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            pass
        
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)#https://github.com/fadel/pytorch_ema
        
        __temp, unit_name = self.scale_the_metrics_(0.,name_run)
        
        for g in optimizer.param_groups:
            #g['lr'] = 0.0
            print({'learning rate': g['lr']})
            #print({'learning rate': g['lr']})
        
        best_valid = float('inf')
        best_test = float('inf')
        
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        path_csv_0 = os.path.join(save_dir, save_dir[len(str("./chk_files/")):]+"_graph_result.csv")
        #csv_filename = 'training_metrics.csv'
        
        #lst_csv_0 = ['epoch', 'training_MAE - '+str(unit_name), 'validation_MAE - '+str(unit_name), 'test_MAE - '+str(unit_name), "Params", "Seed"]
        lst_csv_0 = ['epoch', 'training_MAE - '+str(unit_name), 'validation_MAE - '+str(unit_name), 'test_MAE - '+str(unit_name), "Params", "Seed"]
        #header = ['epoch', 'loss', 'accuracy']
        
        
        csv_logger = CustomLogger(path_csv_0, lst_csv_0)
        
        
        for epoch in range(1, epochs + 1):
            
            print("\n=====Epoch {}".format(epoch), flush=True)
            
            #for g in optimizer.param_groups:#y = 2E-05x - 2E-05
                #print({'learning rate': g['lr']})
                
            print({"#Epoch - "+str(name_run): float(epoch)})
            print('\nTraining...', flush=True)
            train_mae = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device, mean_train , std_train, epoch, scheduler)
            train_mae, unit_name = (train_mae,"kcal/mol")#self.scale_the_metrics_(train_mae,name_run)
            
            #ema.update()
            
            ###Normal weights
            print('\n\nEvaluating...', flush=True)
            type_label = "Valid"
            valid_mae = self.val(model, valid_loader, energy_and_force, p, evaluation, device, mean_train , std_train, type_label)
            valid_mae, unit_name = (valid_mae,"kcal/mol")#self.scale_the_metrics_(valid_mae,name_run)
            
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                #best_test = test_mae
                if save_dir != '':
                    print('Saving checkpoint...')
                    delete_past_chk(save_dir)
                    #checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint_epoch_'+str(epoch)+'.pt'))
            
            
            
            test_now = False
            
            
            if epoch == num_epochs:#150
                print('\n\nTesting...', flush=True)
                #PATH = use_chk_path
                #print(save_dir)
                #print(find_pt_file(save_dir))
                PATH = save_dir+"/"+find_pt_file(save_dir)
                #time.sleep(999)
                print(PATH)
                checkpoint = torch.load(PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                type_label = "Test"
                test_mae = self.val(model, test_loader, energy_and_force, p, evaluation, device, mean_train , std_train, type_label)
                test_mae, unit_name = (test_mae,"kcal/mol")#self.scale_the_metrics_(test_mae,name_run)
                print({'final_test_mae - '+str(unit_name): test_mae})
                #time.sleep(999)
            else:
                test_mae = 699999.0
                unit_name = "kcal/mol"
                #print({'final_test_mae - '+str(unit_name): test_mae})
            
            ###EMA weights
            #with ema.average_parameters():
            #    print('\n\nEvaluating_EMA...', flush=True)
            #    valid_mae_EMA = self.val(model, valid_loader, energy_and_force, p, evaluation, device, mean_train , std_train)
            #    valid_mae_EMA, unit_name = (train_mae,"kcal/mol")#self.scale_the_metrics_(valid_mae_EMA,name_run)
            #    
            #    print('\n\nTesting_EMA...', flush=True)
            #    test_mae_EMA = self.val(model, test_loader, energy_and_force, p, evaluation, device, mean_train , std_train)
            #    test_mae_EMA, unit_name = (train_mae,"kcal/mol")#self.scale_the_metrics_(test_mae_EMA,name_run)
            
            
            
            
            print()
            #print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})

            if log_dir != '':
                #writer.add_scalar('train_mae - '+str(unit_name), train_mae, epoch)
                #writer.add_scalar('valid_mae - '+str(unit_name), valid_mae, epoch)
                #writer.add_scalar('test_mae - '+str(unit_name), test_mae, epoch)
                #print({'train_mae - '+str(unit_name): train_mae})
                #print({'valid_mae - '+str(unit_name): valid_mae})
                #print({'test_mae - '+str(unit_name): test_mae})
                csv_logger.log([epoch, train_mae, valid_mae, test_mae, num_params, seed_id_for_record])
            
                
            #elif valid_mae_EMA < best_valid:
            #    best_valid = valid_mae_EMA
            #    best_test = test_mae_EMA
            #    if save_dir != '':
            #        print('Saving checkpoint...')
            #        delete_past_chk(save_dir)
            #        #checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
            #        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
            #        torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint_epoch_'+str(epoch)+'.pt'))
            #
            #scheduler.step()
            
            
        
        print(f'Best validation MAE so far: {best_valid}')
        print({'Best validation MAE so far': best_valid})
        print(f'Test MAE when got best validation result: {best_test}')
        print({'Test MAE when got best validation result': best_test})
        
        if log_dir != '':
            writer.close()

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device, mean_train , std_train, epoch, scheduler):
        r"""
        The script for training.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.

        :rtype: Traning loss. ( :obj:`mae`)
        
        """   
        
        
        model.train()
        
        loss_accum = 0
        add_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            batch_data = batch_data.to(device)
            #print(batch_data)
            #time.sleep(999)
            #print(step)
            #batch_data.set_data("temp_",torch.rand(25))
            #print("batch_data.shape: "+str(batch_data.append(-1)))
            #time.sleep(999)
            #aux_log = torch.tensor([[0,0]]).cuda()
            
            #input_data = (batch_data, aux_log)
            out, aux_loss, opt_aux = model(batch_data)
            #out = model(batch_data)
            #print(out.shape)
            
            
            
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss + p * f_loss
                #loss = (0.2*e_loss) + (0.8 * f_loss)
                loss_inner = loss + aux_loss
                
                #print({})
                #print({})
                print({'Training total loss per step': loss, 'Training energy loss per step': e_loss, 'Training force loss per step': f_loss})
                if opt_aux == True:
                    print({'Aux loss per step': aux_loss})
                
                
            else:
                #print(batch_data.y.unsqueeze(1).shape)
                #time.sleep(999)
                loss_inner = loss_func(out.cuda(), batch_data.y.unsqueeze(1).cuda()) + aux_loss
                loss = loss_func(out.cuda(), batch_data.y.unsqueeze(1).cuda())# + aux_loss
                
                #print(batch_data.y)
                #time.sleep(999)
                print({'Training loss per step': loss})
                if opt_aux == True:
                    print({'Aux loss per step': aux_loss})
            
            accum_iter = 1#2
            
            loss = loss / accum_iter
            loss_inner = loss_inner / accum_iter
            
            
            
            loss_inner.backward()
            
            
            
            
            if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                add_accum += 1
            
            
            for g in optimizer.param_groups:
                print({'learning rate': g['lr']})
                #print({'learning rate': g['lr']})
            
            scheduler.step()
            
            
            loss_accum += loss.detach().cpu().item()
            
        
        #print({'Valid Energy MAE': energy_mae, 'Valid Force MAE': force_mae})
        
        return loss_accum/(add_accum)#loss_accum / (step + 1)

    def val(self, model, data_loader, energy_and_force, p, evaluation, device, mean_train , std_train, type_label):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()
        
        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)
        
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out, aux_loss, opt_aux = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                preds_force = torch.cat([preds_force,force.detach_()], dim=0)
                targets_force = torch.cat([targets_force,batch_data.force], dim=0)
            preds = torch.cat([preds, out.detach()], dim=0)#preds = torch.cat([preds, out.detach_()], dim=0)#preds = torch.cat([preds.to("cuda"), out.to("cpu")], dim=0)
            sec_t = batch_data.y.unsqueeze(1)
            targets = torch.cat([targets.to("cuda"), sec_t.cuda()], dim=0)#targets = torch.cat([targets.to("cpu"), batch_data.y.unsqueeze(1).to("cpu")], dim=0)

        input_dict = {"y_true": targets, "y_pred": preds}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            if type_label == "Valid":
                print({'Valid Energy MAE': energy_mae, 'Valid Force MAE': force_mae})
                print({'Valid energy loss per step': energy_mae, 'Valid force loss per step': force_mae})
            elif type_label == "Test":
                print({'Final Test Energy MAE': energy_mae, 'Final Test Force MAE': force_mae})
                #print("sss")
                #print("sss")
                #print("sss")
                #print("sss")
                #print("sss")
                #print({'Test energy loss per step': energy_mae, 'Test force loss per step': force_mae})
            
            return energy_mae + p * force_mae

        return evaluation.eval(input_dict)['mae']#, energy_mae, force_mae
