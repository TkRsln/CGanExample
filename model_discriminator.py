
"""
Created on Sun May 14 17:00:07 2023

@author: utkua
"""
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,device=None ,image_dim=28*28, label_dim=10,train_id=None):
        super(Discriminator, self).__init__()
        self.image_dim = image_dim
        self.label_dim = label_dim

        self.model = nn.Sequential(
            nn.Linear(image_dim + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        
        if train_id != None:
            self.load_params(train_id)

    def forward(self, image, labels):
        x = torch.cat((image, labels), dim=1)
        validity = self.model(x)
        return validity
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/discriminator_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/discriminator_{str(train_id)}"))
        self.eval()
        
