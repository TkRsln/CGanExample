"""
Created on Mon May 15 23:16:55 2023

@author: utkua
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import matplotlib.pyplot as plt


def draw_1(generator):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate some samples
    num_samples = generator.label_dim
    sample_noise = torch.randn(num_samples, generator.latent_dim).to(device)
    sample_labels = torch.eye(generator.label_dim).to(device)
    generated_samples = generator(sample_noise, sample_labels).detach().cpu()

    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    
    for i in range(num_samples):
        axes[i].imshow(generated_samples[i].view(28, 28), cmap="gray")
        axes[i].axis("off")
    
    plt.show()
    return plt


def draw_2(generator,lst=[1,9,9,9]):
    #label_dim=10, image_dim= 28*28, latent_dim=100,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate some samples
    num_samples = len(lst)
    sample_noise = torch.randn(num_samples, generator.latent_dim).to(device)
    #sample_labels = torch.eye(n=num_samples,m=generator.label_dim,out=lst).to(device)
    
    sample_labels = torch.zeros(num_samples, 10)
    for i,v in enumerate(lst):
        sample_labels[i][v]=1
    
    generated_samples = generator(sample_noise, sample_labels).detach().cpu()

    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    
    for i in range(num_samples):
        axes[i].imshow(generated_samples[i].view(28, 28), cmap="gray")
        axes[i].axis("off")
    
    plt.show()
    return plt


def draw_3(generator,lst=[1,9,9,9]):
    #label_dim=10, image_dim= 28*28, latent_dim=100,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate some samples
    num_samples = len(lst)
    sample_noise = torch.randn(num_samples, generator.latent_dim).to(device)
    #sample_labels = torch.eye(n=num_samples,m=generator.label_dim,out=lst).to(device)
    
    sample_labels = torch.zeros(num_samples, 10)
    for i,v in enumerate(lst):
        sample_labels[i][v]=1
    
    generated_samples = generator(sample_noise, sample_labels).detach().cpu()
    shaped_sample=[]
    for gs in generated_samples:
        shaped_sample.append(gs.view(28,28))
    

    merged = torch.cat(shaped_sample,dim=1)
    # Plot the image
    plt.imshow(merged.view(28,28*num_samples), cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

def draw_4(generator,lst='30 01 1999',show=False):
    #label_dim=10, image_dim= 28*28, latent_dim=100,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    shaped_sample=[]
    for s in lst:           
        if s ==' ':
            shaped_sample.append(torch.ones(28, 28)*-1)
        else:
            s=int(s)
            sample_noise = torch.randn(1,generator.latent_dim).to(device)        
            sample_labels = torch.zeros(1,10)
            sample_labels[0][s]=1
            gen=generator(sample_noise, sample_labels).detach().cpu()
            shaped_sample.append(gen.view(28,28))
    
    num_samples=len(shaped_sample)
    merged = torch.cat(shaped_sample,dim=1)
    # Plot the image
    plt.imshow(merged.view(28,28*num_samples), cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('./AI_3_OUT.png')

"""
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    
    for i in range(num_samples):
        axes[i].imshow(generated_samples[i].view(28, 28), cmap="gray")
        axes[i].axis("off")
    
    plt.show()
    return plt
"""
    




def train(generator:nn.Module,discriminator:nn.Module,adversarial_loss=nn.BCELoss(),lr = 0.0002,num_epochs=25,batch_size = 64,image_dim = 28 * 28):
    # Hyperparameters
    latent_dim = generator.latent_dim
    label_dim =generator.label_dim #10
    #num_epochs = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    
    #adversarial_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    
    total_process=num_epochs*len(dataloader)
    bar = tqdm(total=total_process,desc='starting...')
    bar_update=5
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.view(-1, image_dim).to(device)
            labels = torch.eye(label_dim)[labels].to(device)  # One-hot encoding
            
            # Train the DISC
            discriminator_optimizer.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
    
            real_validity = discriminator(real_images, labels)
            real_loss = adversarial_loss(real_validity, real_labels)
    
            noise = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(noise, labels)
            fake_validity = discriminator(fake_images, labels)
            fake_loss = adversarial_loss(fake_validity, fake_labels)
    
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()
    
            # Train the GEN
            generator_optimizer.zero_grad()
            noise = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(noise, labels)
            fake_validity = discriminator(fake_images, labels)
            generator_loss = adversarial_loss(fake_validity, real_labels)
    
            generator_loss.backward()
            generator_optimizer.step()
    
            # INFO
            if (i + 1) % bar_update == 0:
                bar.update(bar_update)
                bar.desc=f"Epoch:[{epoch+1}/{num_epochs}-({i+1}/{len(dataloader)})], Loss:[D:{discriminator_loss.item():.4f}, G:{generator_loss.item():.4f}]"
                
                """
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                    f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                    f"Generator Loss: {generator_loss.item():.4f}"
                )
                """
