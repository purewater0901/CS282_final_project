from deep_learning.full_fine_tuner import FullFT
from torch import optim, nn
import torch
import numpy as np
from tqdm.auto import tqdm
import gc
import matplotlib.pyplot as plt
import os

class BayesTune(FullFT):
    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate = None,
        model=None,
        processor=None,
        device = "cpu",
        **kwargs
    ):
        super().__init__(
            model_name,
            data_dir,
            real_folder,
            fake_folder,
            num_epochs,
            batch_size,
            learning_rate,
            model,
            processor,
            device,
        )
        self.method_name = 'Bayes_FT'
        self.lambda_stats = None
        self.count_total_params = sum(p.numel() for p in self.model.parameters())

        # Get hyperparameters for Bayesian Inference
        self.sparsity_level = kwargs.get('sparsity_level')
        self.lr_model= kwargs.get('lr_model')
        self.lr_lambda= kwargs.get('lr_lambda')
        self.num_epochs_sampling= kwargs.get('num_epochs_sampling')
        self.N_exp= kwargs.get('N_exp')
        self.burnin_steps= kwargs.get('burnin_steps')
        self.thinning_steps= kwargs.get('thinning_steps')
        self.warmup_steps= kwargs.get('warmup_steps')
        self.batch_size_sampling = kwargs.get('batch_size_sampling')
        
        # Get all parameter names from the model
        self.group_names = [name for name, _ in self.model.named_parameters()]
        print(f"Found {len(self.group_names)} parameter groups, with {self.count_total_params} total parameters")
        
        # Store initial parameter values
        self.initial_params = {}
        for name, param in self.model.named_parameters():
            self.initial_params[name] = param.clone().detach()

        torch.save(self.model.state_dict(), "bayes_tune_initial_model.pt")


    def get_posterior_stats(
        self,
        alpha=0.01,
        beta=100,
        lambda_init=0.0001,
        noise_discount=1.0,
        ):
        
        # Clear CUDA cache
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Move model to the device
        self.model = self.model.to(self.device)

        # Unfreeze all layers
        self.unfreeze_all_layers()
                
        # Initialize lambda parameter for each group - one scalar per parameter group
        self.lambda_params = nn.ParameterList([
            nn.Parameter(lambda_init * torch.ones(1).to(self.device))
            for _ in range(len(self.group_names))
        ]).to(self.device)
        
        # Create mapping from group name to lambda parameter index
        # May not be necessary but let's keep it for now
        self.group_to_lambda_idx = {group_name: i for i, group_name in enumerate(self.group_names)}
        
        # Set optimizer for model params and Bayesian params separately
        model_optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_model)
        lambda_optimizer = optim.Adam(self.lambda_params.parameters(), lr=self.lr_lambda)
        
        # Loss function and optimizer -> this makes loss function equivalent to Negative log likelihood
        criterion = nn.CrossEntropyLoss()
        
        # N parameter for SGLD
        N = 10**self.N_exp
        
        # For collecting lambda statistics
        lambda_stats = {'running_mean': None, 'count': 0, 'group_names': self.group_names}
        
        # Training loop
        global_step = 0

        # Get data loader
        train_loader, _ = self.get_Train_Val_loader(custom_batch_size=self.batch_size_sampling)

        # Stochastic Gradient Langevin Dynamics!!!
        print("Sampling with SGLD...")
        for epoch in range(self.num_epochs_sampling):
            self.model.train()
            
            for images, labels, pathes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs_sampling}"):
                global_step += 1 # not counting epoch, but counting stochastic trails
                
                # Move batch to device
                inputs = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass and loss calculation
                if self.model_name == 'deepfake_detection':
                    outputs = self.model(inputs).logits_labels
                else:
                    outputs = self.model(inputs)
                    
                loss = criterion(outputs, labels)  
                
                # Backward pass for model parameters
                model_optimizer.zero_grad()
                loss.backward()
                
                # SGLD update for lambda parameters
                is_warmup = global_step < self.warmup_steps
                
                if not is_warmup:
                    lambda_optimizer.zero_grad()
                    
                    # Use clamp to make sure the lambda is always positive
                    lambda_pos = [torch.clamp(l, min=1e-8) for l in self.lambda_params]
                    
                    # Update model parameters and compute lambda gradients
                    for idx, (name, param) in enumerate(self.model.named_parameters()):
                        # Get lambda for this parameter group - lambda is a scalar now
                        lambda_val = lambda_pos[idx]
                        
                        # Gradient of Gamma Prior:
                        lambda_grad = -((alpha-1)/lambda_val - beta) / N
                        
                        # Likelihood gradient term (Laplace Prior)
                        init_param = self.initial_params[name]
                        param_diff = param - init_param
                        
                        # Compute gradient for this lambda
                        lambda_grad += -1 * ((param_diff.abs().sum()/(lambda_val**2) - param_diff.numel()/lambda_val)) / N
                        
                        # Add noise term for Langevin dynamics
                        noise = np.sqrt(2/(N*self.lr_lambda)) * torch.randn_like(lambda_val) * noise_discount
                        lambda_grad += noise
                        
                        # Set gradient
                        self.lambda_params[idx].grad = lambda_grad
                        
                        # Add L1 regularization to model parameters based on lambda
                        if param.grad is not None:
                            param.grad += -(-torch.sign(param_diff) / lambda_val) / N
                            
                            # Add noise to model gradients for Langevin dynamics
                            if self.lr_model > 0:
                                param.grad += np.sqrt(2/(N*self.lr_model)) * torch.randn_like(param) * noise_discount
                    
                    # Update lambda parameters
                    lambda_optimizer.step()
                    
                    # Collect lambda statistics after burn-in period
                    if global_step > self.burnin_steps and global_step % self.thinning_steps == 0:
                        lambda_vec = torch.cat([l.flatten() for l in lambda_pos])
                        
                        if lambda_stats['count'] == 0:
                            lambda_stats['running_mean'] = lambda_vec.clone()
                        else:
                            lambda_stats['running_mean'] = (lambda_vec + lambda_stats['count']*lambda_stats['running_mean']) / (lambda_stats['count']+1)
                        
                        lambda_stats['count'] += 1

                # Update model parameters
                model_optimizer.step()

                # Clear CUDA cache after each batch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                gc.collect()

        # Log lambda statistics at the end of the epoch
        with torch.no_grad():
            print('Sampling completed;')
            print(f"Lambda stats - Mean: {lambda_stats['running_mean'].mean().item():.6f}, "
                  f"Min: {lambda_stats['running_mean'].min().item():.6f}, "
                  f"Max: {lambda_stats['running_mean'].max().item():.6f}, "
                  f"Median: {lambda_stats['running_mean'].median().item():.6f}")
            
            # Print top 10 parameters with highest lambda values
            
            # Visualize parameter group mean
            plt.plot(lambda_stats['running_mean'].detach().cpu().numpy())
            plt.title(" Importance per Parameter Group")
            plt.xlabel("Parameter Group Index")
            plt.ylabel("Lambda (Importance)")
            plt.savefig(f"/home/ec2-user/CS282_final_project/src/pics/lambda_mean.png")

        # Clear CUDA cache
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()
        self.lambda_stats = lambda_stats
        
        # Save lambda statistics with group mapping
        torch.save({
            'running_mean': lambda_stats['running_mean'],
            'group_names': self.group_names
        }, "bayes_tune_lambda_stats.pt")

        return lambda_stats
        

    def Tune(self):
        # Get posterior stats
        stats_file = "bayes_tune_lambda_stats.pt"
        if os.path.exists(stats_file):
            print(f"Loading lambda stats from {stats_file}")
            lambda_stats = torch.load(stats_file, map_location=self.device)
            lambda_mean = lambda_stats['running_mean'].detach().cpu().numpy()
        else:
            lambda_stats = self.get_posterior_stats()
            lambda_mean = lambda_stats['running_mean'].detach().cpu().numpy()

        # Reload the initial model for the second stage fine-tuning
        self.model.load_state_dict(torch.load("bayes_tune_initial_model.pt", map_location=self.device))
        
        sorted_values, sorted_indices = torch.tensor(lambda_mean).sort(descending=True)
        total_selected = 0
        
        # Set the top p% of the parameters by group mean to be trainable
        print("Selecting parameters to be trainable...")

        # Freeze all layers first
        self.freeze_all_layers()

        # Selectively unfreeze
        for idx in sorted_indices:
            if (total_selected/self.count_total_params) <= self.sparsity_level:
                params = self.model.get_parameter(self.group_names[idx])
                params.requires_grad = True
                total_selected += params.numel()
                #print(f"Group name: {self.group_names[idx]} ({params.numel()} params), Total: {total_selected}/{self.count_total_params} ({total_selected/self.count_total_params*100:.4f}%)")
            else:
                break

        # Show total updatable parameters
        print(f"Stop accepting at: {total_selected}/{self.count_total_params} ({total_selected/self.count_total_params*100:.4f}%)")

        # Check if the count matches
        assert self.count_trainable_params() == total_selected, "Trainable parameters count doesn't match selected parameters count"

        model_save_path = f"bayes_tune_{self.model_name}.pth"

        return super().Tune(model_save_path=model_save_path)

