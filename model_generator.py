# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:00:07 2023

@author: utkua
"""
import torch
import torch.nn as nn

# Define the generator network
class Generator(nn.Module):
    def __init__(self,device=None, label_dim=10, image_dim= 28*28, latent_dim=100,train_id=None):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.image_dim = image_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        if train_id != None:
            self.load_params(train_id)
        

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        generated_image = self.model(x)
        return generated_image

    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/generator_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/generator_{str(train_id)}"))
        self.eval()