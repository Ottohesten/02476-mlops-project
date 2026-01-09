"""Adapted from https://github.com/Jackson-Kang/PyTorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import logging
import os

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from mlops_project.model_week3 import Decoder, Encoder, Model

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config.yaml")
def train(config):
    """Train VAE on MNIST dataset."""
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    config = config

    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(config.seed)

    # Data loading
    # mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(config.dataset_path, train=True, download=True)
    train_dataset = TensorDataset(train_dataset.data.float() / 255.0, train_dataset.targets)

    test_dataset = MNIST(config.dataset_path, train=False, download=True)
    test_dataset = TensorDataset(test_dataset.data.float() / 255.0, test_dataset.targets)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)
    encoder = Encoder(input_dim=config.x_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
    decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, output_dim=config.x_dim)

    model = Model(encoder=encoder, decoder=decoder).to(DEVICE)

    def loss_function(x, x_hat, mean, log_var):
        """Elbo loss function."""
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld

    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    # optimizer = Adam(model.parameters(), lr=config.lr)

    log.info("Start training VAE...")
    model.train()
    for epoch in range(config.num_epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(config.batch_size, config.x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch + 1} complete!,  Average Loss: {overall_loss / (batch_idx * config.batch_size)}")
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                log.info(f"Processing batch {batch_idx} for reconstruction")
            x = x.view(config.batch_size, config.x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(config.batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(config.batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(config.batch_size, 20).to(DEVICE)
        generated_images = decoder(noise)

    save_image(generated_images.view(config.batch_size, 1, 28, 28), "generated_sample.png")


if __name__ == "__main__":
    train()
