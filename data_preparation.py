import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision as tv

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import normflows as nf

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def create_glow_blocks(L, K, channels, hidden_channels, split_mode, scale):
    """
    Create Glow blocks for the flow model.
    """
    flows = []
    merges = []
    
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                          split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows.append(flows_)
        if i > 0:
            merges.append(nf.flows.Merge())
    
    return flows, merges

def create_latent_distributions(L, input_shape, num_classes):
    """
    Create latent distributions for the flow model.
    """
    q0 = []
    
    for i in range(L):
        if i > 0:
            latent_shape = (
                input_shape[0] * 2 ** (L - i),
                input_shape[1] // 2 ** (L - i),
                input_shape[2] // 2 ** (L - i)
            )
        else:
            latent_shape = (
                input_shape[0] * 2 ** (L + 1),
                input_shape[1] // 2 ** L,
                input_shape[2] // 2 ** L
            )
        q0.append(nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes))
    
    return q0

def create_multiscale_flow(L, K, input_shape, hidden_channels, split_mode, scale, num_classes):
    """
    Create the multiscale flow model.
    """
    channels = input_shape[0]
    flows, merges = create_glow_blocks(L, K, channels, hidden_channels, split_mode, scale)
    q0 = create_latent_distributions(L, input_shape, num_classes)
    
    return nf.MultiscaleFlow(q0, flows, merges)

class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        """
        Initialize the dataset by scanning the folder for images in 'NORMAL' and 'PNEUMONIA' subdirectories.
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load NORMAL images with label 0
        normal_folder = os.path.join(folder, 'NORMAL')
        self.image_paths += [
            os.path.join(normal_folder, fname)
            for fname in os.listdir(normal_folder)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.labels += [0] * len(os.listdir(normal_folder))

        # Load PNEUMONIA images with label 1
        pneumonia_folder = os.path.join(folder, 'PNEUMONIA')
        self.image_paths += [
            os.path.join(pneumonia_folder, fname)
            for fname in os.listdir(pneumonia_folder)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.labels += [1] * len(os.listdir(pneumonia_folder))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load an image and its corresponding label.
        """
        img = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # You can increase this if memory allows
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),  # Ensure the images are grayscale
    nf.utils.Jitter(1 / 256.),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_with_loop(model, train_loader, device, epochs=100, lr=1e-3, weight_decay=1e-5):
    """
    Train the model using a standard loop over the DataLoader.
    """
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()  # Set the model to training mode
        iterat = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            with torch.enable_grad():  # Ensure gradients are enabled
                loss = model.forward_kld(x, y)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                loss_hist.append(loss.detach().cpu().numpy())
            else:
                print("Skipping NaN or Inf loss.")

            # iterat += 1
            # if iterat % 5 == 0:
            #     model.eval()  # Set the model to evaluation mode for log_prob calculation
            #     with torch.no_grad():
            #         log_prob = model.log_prob(x, y)
            #     print(f"Iteration {iterat}, Loss: {loss.item()}, Log Prob: {log_prob.mean().item()}")
            #     model.train()  # Switch back to training mode

        print(f"Epoch {epoch + 1} completed. Loss: {loss_hist[-1] if loss_hist else 'N/A'}")

    return loss_hist

def plot_loss(loss_hist, save_path="training_loss.png"):
    """
    Plot the training loss history and save the figure.
    """
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def generate_samples(model, num_samples, num_classes, device, save_dir="./generated_xrays"):
    """
    Generate and save samples from the model with labels.
    """
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        y = torch.arange(num_classes).repeat(num_samples).to(device)
        x, _ = model.sample(y=y)
        x_ = torch.clamp(x, 0, 1)
        grid = tv.utils.make_grid(x_, nrow=num_classes)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title("Generated Samples: Normal and Pneumonia")
        plt.savefig(os.path.join(save_dir, "generated_samples.png"))
        plt.close()

def calculate_bits_per_dim(model, test_loader, device, n_dims):
    """
    Calculate bits per dimension (BPD) for the model.
    """
    n = 0
    bpd_cum = 0
    with torch.no_grad():
        for x, y in iter(test_loader):
            nll = model(x.to(device), y.to(device))
            nll_np = nll.cpu().numpy()
            bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
            n += len(x) - np.sum(np.isnan(nll_np))
    bpd = bpd_cum / n
    print('Bits per dim: ', bpd)
    return bpd

if __name__ == "__main__":

    L = 5
    K = 16

    input_shape = (1, 64, 64)
    n_dims = np.prod(input_shape)
    hidden_channels = 256
    channels = 1
    split_mode = 'channel'
    scale = True
    num_classes = 2

    model = create_multiscale_flow(
        L=L,
        K=K,
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        split_mode=split_mode,
        scale=scale,
        num_classes=num_classes
    ).to(device)

    epochs = 200

    normal_dataset = ChestXrayDataset(
        folder='dataset/Pediatric Chest X-ray Pneumonia/train',
        transform=transform
    )
    normal_loader = DataLoader(normal_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=18)

    # model.load_state_dict(torch.load('multiscale_flow_model.pth', map_location=device))
    # print("Model loaded from 'multiscale_flow_model.pth'")

    # Train the model
    loss_hist = train_model_with_loop(model, normal_loader, device=device, epochs=epochs)
    print("Training completed.")
    
    # Plot the training loss
    plot_loss(loss_hist)
    
    num_samples = 3
    # Generate samples
    generate_samples(model, num_samples, num_classes, device)
    
    # Calculate bits per dimension
    test_loader = DataLoader(normal_dataset, batch_size=128, shuffle=False)
    bpd = calculate_bits_per_dim(model, test_loader, device, n_dims)
    print(f"Bits per dimension: {bpd}")
    # Save the model
    torch.save(model.state_dict(), 'multiscale_flow_model_l5.pth')
    print("Model saved as 'multiscale_flow_model.pth'")