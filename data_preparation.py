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
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform

from nflows.flows import Flow

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    """A standard multi-layer perceptron compatible with nflows."""

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
    ):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, *args, **kwargs):
        # nflows may pass multiple arguments; extract the actual input
        inputs = args[0]
        
        if inputs.shape[1:] != self._in_shape:
            raise ValueError(
                f"Expected inputs of shape {self._in_shape}, got {inputs.shape[1:]}."
            )

        inputs = inputs.view(-1, np.prod(self._in_shape))
        outputs = self._input_layer(inputs)
        outputs = self._activation(outputs)

        for layer in self._hidden_layers:
            outputs = layer(outputs)
            outputs = self._activation(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)

        outputs = outputs.view(-1, *self._out_shape)
        return outputs



class ChestXrayNormalDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.image_paths = [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # You can increase this if memory allows
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten
])

normal_dataset = ChestXrayNormalDataset(
    folder='dataset/Pediatric Chest X-ray Pneumonia/train/NORMAL',
    transform=transform
)
normal_loader = DataLoader(normal_dataset, batch_size=64, shuffle=True)


def create_flow(dim, hidden_sizes):
    def create_net(input_dim, output_dim):
        return MLP(
            in_shape=(input_dim,), 
            out_shape=(output_dim,), 
            hidden_sizes=hidden_sizes,
            activation=F.relu,
            activate_output=False
        )

    transform = CompositeTransform([
        RandomPermutation(features=dim),
        PiecewiseRationalQuadraticCouplingTransform(
        mask=torch.arange(dim) % 2,
        transform_net_create_fn=create_net,
        tails="linear",  # Allow unbounded extrapolation
        tail_bound=1.0,  # Assumes input lies in [-1, 1] or [0, 1]
        num_bins=8,  # Can reduce if you want simpler modeling
        apply_unconditional_transform=False
        )
    ])
    
    base_dist = StandardNormal([dim])
    return Flow(transform, base_dist)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dim = 64 * 64
flow = create_flow(dim=image_dim, hidden_sizes=[256, 256]).to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
epochs = 100

for epoch in range(epochs):
    flow.train()
    total_loss = 0
    for x in tqdm(normal_loader, desc=f"Epoch {epoch+1}"):
        x = x.to(device)
        x = x.clamp(1e-4, 1.0 - 1e-4)  # Clamping to avoid log(0)
        loss = -flow.log_prob(x).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(normal_loader):.4f}")


num_samples = 10  # or any number you like

# Output directory
save_dir = "./generated_xrays"
os.makedirs(save_dir, exist_ok=True)

# Sampling
flow.eval()
with torch.no_grad():
    samples = flow.sample(num_samples).view(-1, 1, 64, 64).cpu()
    for i in range(num_samples):
        img = transforms.ToPILImage()(samples[i])
        img.save(os.path.join(save_dir, f"normal_gen_{i}.png"))