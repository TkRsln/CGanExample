# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:42:05 2023

@author: utkua
"""

import train_test as tool
from model_discriminator import Discriminator
from model_generator import Generator


gen = Generator(train_id=200)
dis = Discriminator(train_id=200)

tool.draw_4(gen,'99 01 1999',show=True)
#tool.train(gen, dis,num_epochs=2)