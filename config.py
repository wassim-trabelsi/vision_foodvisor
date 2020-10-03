import torch
from model import vgg_model
import torch.nn as nn

TomateSolide = ['Tomates séchées','Tomate Mozzarella (Salade Caprese plat complet)','Tartine fromage tomate jambon',
          'Tomate farcie', 'Tomate à la provençale', 'Tomates (entières)', 'Tomates cerises', 'Tomates (coupées)'
              ]

# Choose the hyperparameters for training:
num_epochs = 50
batch_size = 5

# Training criterion.
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 5.]).to('cuda'))

optimizer = torch.optim.Adam(vgg_model.parameters(), lr=1e-4)

