# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 04:53:24 2023

@author: varsh
"""
#https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/

# Importing the necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torch 
plt.rcParams['figure.figsize'] = 15, 10

# Initializing the transform for the dataset 
transform = torchvision.transforms.Compose([ 
	torchvision.transforms.ToTensor(), 
	torchvision.transforms.Normalize((0.5), (0.5)) 
]) 

# Downloading the MNIST dataset 
train_dataset = torchvision.datasets.MNIST( 
	root="./MNIST/train", train=True, 
	transform=torchvision.transforms.ToTensor(), 
	download=True) 

test_dataset = torchvision.datasets.MNIST( 
	root="./MNIST/test", train=False, 
	transform=torchvision.transforms.ToTensor(), 
	download=True) 

# Creating Dataloaders from the 
# training and testing dataset 
train_loader = torch.utils.data.DataLoader( 
	train_dataset, batch_size=256) 
test_loader = torch.utils.data.DataLoader( 
	test_dataset, batch_size=256) 

# Printing 25 random images from the training dataset 
random_samples = np.random.randint( 
	1, len(train_dataset), (25)) 

for idx in range(random_samples.shape[0]): 
	plt.subplot(5, 5, idx + 1) 
	plt.imshow(train_dataset[idx][0][0].numpy(), cmap='gray') 
	plt.title(train_dataset[idx][1]) 
	plt.axis('off') 

plt.tight_layout() 
plt.show() 

# Creating a DeepAutoencoder class 
class DeepAutoencoder(torch.nn.Module): 
	def __init__(self): 
		super().__init__()		 
		self.encoder = torch.nn.Sequential( 
			torch.nn.Linear(28 * 28, 256), 
			torch.nn.ReLU(), 
			torch.nn.Linear(256, 128), 
			torch.nn.ReLU(), 
			torch.nn.Linear(128, 64), 
			torch.nn.ReLU(), 
			torch.nn.Linear(64, 10) 
		) 
		
		self.decoder = torch.nn.Sequential( 
			torch.nn.Linear(10, 64), 
			torch.nn.ReLU(), 
			torch.nn.Linear(64, 128), 
			torch.nn.ReLU(), 
			torch.nn.Linear(128, 256), 
			torch.nn.ReLU(), 
			torch.nn.Linear(256, 28 * 28), 
			torch.nn.Sigmoid() 
		) 

	def forward(self, x): 
		encoded = self.encoder(x) 
		decoded = self.decoder(encoded) 
		return decoded 

# Instantiating the model and hyperparameters 
model = DeepAutoencoder() 
criterion = torch.nn.MSELoss() 
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 


# List that will store the training loss 
train_loss = [] 

# Dictionary that will store the 
# different images and outputs for 
# various epochs 
outputs = {} 

batch_size = len(train_loader) 

# Training loop starts 
for epoch in range(num_epochs): 
		
	# Initializing variable for storing 
	# loss 
	running_loss = 0
	
	# Iterating over the training dataset 
	for batch in train_loader: 
			
		# Loading image(s) and 
		# reshaping it into a 1-d vector 
		img, _ = batch 
		img = img.reshape(-1, 28*28) 
		
		# Generating output 
		out = model(img) 
		
		# Calculating loss 
		loss = criterion(out, img) 
		
		# Updating weights according 
		# to the calculated loss 
		optimizer.zero_grad() 
		loss.backward() 
		optimizer.step() 
		
		# Incrementing loss 
		running_loss += loss.item() 
	
	# Averaging out loss over entire batch 
	running_loss /= batch_size 
	train_loss.append(running_loss) 
	
	# Storing useful images and 
	# reconstructed outputs for the last batch 
	outputs[epoch+1] = {'img': img, 'out': out} 


# Plotting the training loss 
plt.plot(range(1,num_epochs+1),train_loss) 
plt.xlabel("Number of epochs") 
plt.ylabel("Training Loss") 
plt.show()
