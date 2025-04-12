from deep_learning.full_fine_tuner import FullFT
from torch import optim, nn
import torch
import numpy as np
from tqdm.auto import tqdm
import gc
from torch.amp import autocast, GradScaler

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
        sparsity_level = 0.05,
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
        self.sparsity_level = sparsity_level

    def unfreeze_selected_layers(self):
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layornorm layers
        selected_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    selected_layers.append((f"{name}.{param_name}", param))
            
        # Count total parameters and linear parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        selected_params = sum(p.numel() for _, p in selected_layers)
        
        print(f"Enabled {len(selected_layers)} parameter groups with {selected_params} parameters")
        print(f"Total model parameters: {total_params}")
        print(f"Percentage of parameters enabled: {selected_params/total_params*100:.2f}%")
        
        return selected_layers


    def get_posterior_stats(
        self,
        alpha=0.01,
        beta=100,
        lambda_init=0.0001,
        lr_model=2e-5,
        lr_lambda=0.01,
        num_epochs=1,
        N_exp=12,
        noise_discount=1.0,
        burnin_steps=100,
        thinning_steps=10,
        warmup_steps=50):
        

        # Clear CUDA cache
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Move model to the device
        self.model = self.model.to(self.device)

        # Parameter shapes and names
        self.param_shapes = []
        self.trainable_param_names = []

        # Filter trainable parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_shapes.append(list(param.shape))
                self.trainable_param_names.append(name)

        # Create a copy of initial parameters
        self.initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.detach().clone()

        # Initialize lambda parameters only for trainable parameters
        self.lambda_params = nn.ParameterList([
            nn.Parameter(lambda_init * torch.ones(*shape).to(self.device)) 
            for shape in self.param_shapes
        ]).to(self.device)
        
        # Set optimizer for model params and Bayesian params separately
        model_optimizer = optim.AdamW(self.model.parameters(), lr=lr_model)
        lambda_optimizer = optim.Adam(self.lambda_params.parameters(), lr=lr_lambda)
        
        # Loss function and optimizer -> this makes loss function equivalent to Negative log likelihood
        criterion = nn.CrossEntropyLoss()
        
        # N parameter for SGLD
        N = 10**N_exp
        
        # For collecting lambda statistics
        lambda_stats = { 'running_mean': None,'count': 0}
        
        # Training loop
        global_step = 0

        # Get data loader
        train_loader, _ = self.get_Train_Val_loader()

        # Stochastic Gradient Langevin Dynamics!!!
        # SGLD updates both model parameters and regularization parameter lambda, 
        # but this is purely for sampling purposes, not for training
        # 
        for epoch in range(num_epochs):
            self.model.train()
            
            for images, labels, pathes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
                is_warmup = global_step < warmup_steps
                
                if not is_warmup:
                    lambda_optimizer.zero_grad()
                    
                    # Use clamp to make sure the lambda is always positive, otherwise can't interpret Laplace prior as probability density
                    lambda_pos = [torch.clamp(l, min=1e-8) for l in self.lambda_params]
                    
                    # Update both model and regularization parameter together, only for trainable parameters
                    for idx, name in enumerate(self.trainable_param_names):

                        # These are for lambda 
                        # Gradient of Gamma Prior:
                        lambda_grad = -((alpha-1)/lambda_pos[idx] - beta) / N
                        
                        # Likelihood gradient term (Laplace Prior)
                        init_param = self.initial_params[name]
                        param = self.model.get_parameter(name)
                        param_diff = param - init_param
                        lambda_grad += -1 * ((param_diff.abs()/(lambda_pos[idx]**2) - 1/lambda_pos[idx])) / N
                        
                        # Add noise term for Langevin dynamics
                        noise = np.sqrt(2/(N*lr_lambda)) * torch.randn_like(lambda_pos[idx]) * noise_discount
                        lambda_grad += noise
                        
                        # Store gradient
                        self.lambda_params[idx].grad = lambda_grad
                        
                        # Add L1 regularization to model parameters based on lambda
                        if param.grad is not None:
                            param.grad += -(-torch.sign(param_diff) / lambda_pos[idx]) / N
                            
                            # Add noise to model gradients for Langevin dynamics
                            if lr_model > 0:
                                param.grad += np.sqrt(2/(N*lr_model)) * torch.randn_like(param) * noise_discount
                    
                    # Update lambda parameters
                    lambda_optimizer.step()
                    
                    # Collect lambda statistics after burn-in period
                    if global_step > burnin_steps and global_step % thinning_steps == 0:
                        lambda_vec = torch.cat([l.flatten() for l in lambda_pos])
                        
                        if lambda_stats['count'] == 0:
                            lambda_stats['running_mean'] = lambda_vec.clone()
                        else:
                            lambda_stats['running_mean'] = (lambda_vec + lambda_stats['count']*lambda_stats['running_mean']) / (lambda_stats['count']+1)
                        
                        lambda_stats['count'] += 1
                        

                # Update model parameters
                model_optimizer.step()

                # Clear CUDA cache
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                gc.collect()

            # Log lambda statistics at the end of the epoch
            with torch.no_grad():
                print(f"Lambda stats - Mean: {lambda_stats['running_mean'].mean().item():.6f}, "
                      f"Min: {lambda_stats['running_mean'].min().item():.6f}, "
                      f"Max: {lambda_stats['running_mean'].max().item():.6f}, "
                      f"Median: {lambda_stats['running_mean'].median().item():.6f}")
                
        # Clear CUDA cache
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()

        return lambda_stats
        
    def Tune(self):
        # For testing purpose, only target the linear layers

        # Set all the linear layers to be trainable
        self.unfreeze_selected_layers()

        posterior_stats = self.get_posterior_stats()
        lambda_mean = posterior_stats['running_mean']
        print('Lambda Mean Example: ', lambda_mean[:5])

        # If the above works, use that lambda_mean to select which parameters to set requires_grad = True.
        # Create parameter mask based on lambda values (False = update, True = freeze)
        # Sort lambda values and keep top parameters based on sparsity level
        sorted_values, sorted_indices = lambda_mean.sort()
        threshold_idx = int(len(lambda_mean) * self.sparsity_level)
        update_indices = sorted_indices[-threshold_idx:]
        
        print(f"Sparsity level {self.sparsity_level} (updating {threshold_idx} of {len(lambda_mean)} parameters)")
        print(f"Lambda threshold: {sorted_values[-threshold_idx].item():.6f}")

        # Track global index
        global_idx = 0
        updatable_params = []
        total_selected = 0
        
        # Selectively make it trainable
        for name, param in self.model.named_parameters():
            param_size = param.numel()
            
            # Check if any of the indices in the current parameter are in the update_indices
            local_indices = []
            for i in range(param_size):
                if global_idx + i in update_indices:
                    local_indices.append(i)
            
            # If any indices are found, set the chunk of parameters to be trainable            
            if local_indices:
                param.requires_grad = True
                updatable_params.append(param)
                print(f"Parameter {name}: {len(local_indices)}/{param_size} elements updatable ({len(local_indices)/param_size*100:.4f}%)")
                total_selected += len(local_indices)
            # Increment global index
            global_idx += param_size

            if total_selected >= threshold_idx:
                break
        
        # Show total updatable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        updatable_count = sum(p.numel() for p in updatable_params)
        print(f"Total updatable parameters: {updatable_count}/{total_params} ({updatable_count/total_params*100:.4f}%)")
        
        model_save_path = f"bayes_tune_{self.model_name}.pth"

        return super().Tune(model_save_path=model_save_path)


