import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import genModel, discModel, ganModel
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm 



# Function to get real data MNIST samples

def realData(batchsize, digit = None):
	# Get MNIST 
	(x_train, y_train), (_, _) = mnist.load_data()
	if digit != None:
		filter = np.where(y_train == digit)
		x_train, y_train = x_train[filter], y_train[filter]

	x = x_train[:,:,:,np.newaxis]
	x = (x.astype('float32'))
	x = x/255
	# Generating a batch of data
	x_Batch = x[np.random.randint(0, x.shape[0], size=batchsize)]
	return x_Batch
	
# Create random noisy data

def fakeInputData(batchsize,features):
	# Generate random data from noise  
	i_fake = np.random.uniform(-1,1,size=[batchsize,features])
	return i_fake


def fakedataGenerator(Genmodel,batch,features):
	fakeInputs = fakeInputData(batch,features)
	# use these inputs inside the generator model to generate fake distribution
	x_fake = Genmodel.predict(fakeInputs)   
	
	return x_fake



##### Variables

rows = 3
cols = 10
batch = 128
features = 100 # Dose not have to be 768, can be whatever. Just the dimension of the fake noisy input data
nEpochs = 10
reportEpochs = 5 # NUmbers of epochs to report data and get a sample
discriminator = {}

### Generate the models

def train(batch, features, epochs, reportEpochs, digit = None):


	# Generate the models

	#Discriminator
	disc_model = discModel()
	print(disc_model.summary())

	# Generator model
	gen_model = genModel(features)
	print(gen_model.summary())

	#GAN model
	gan_model = ganModel(gen_model,disc_model)
	print(gan_model.summary())

	# Train the GAN models 
	for i in tqdm(range(epochs)):

		################# Disc Part ########################
		# MNIST real data
		if digit == None:
			x_real = realData(batch)
		else:
			x_real = realData(batch, digit)
  		# Generate fake data 
		x_fake = fakedataGenerator(gen_model,batch,features)
		# Dicriminator data
		x_disc = np.concatenate([x_real,x_fake])

		# Creating the label: first half 1 -> real, second half 0 -> fake
		y_disc = np.concatenate([np.ones(batch),np.zeros(batch)])

		# train the  discriminator network with real data
		disc_loss = disc_model.train_on_batch(x_disc, y_disc)

		################# Gan Part ########################

		# Generate fake data
		x_gan = fakeInputData(batch*2,features)
		# Create labels , all set to one to increase losses
		y_gan = np.ones((batch*2, 1))
		# Training gen model 
		gan_model.train_on_batch(x_gan, y_gan)

		# Print the accuracy measures on the real and fake data for every x  epochs
		if (i) % reportEpochs == 0:
			# Printing the descriminator loss and accuracy
			x_test_real = realData(5)
			x_test_fake = fakedataGenerator(gen_model,5,features)
			# Concatenating the real and fake data 
			x_test = np.concatenate([x_test_real,x_test_fake])
			# Creating the dependent variable and initializing them as '0'
			y = np.concatenate([np.ones(5), np.zeros(5)])
			# Predicting probabilities
			preds = disc_model.predict(x_test)
			d = np.mean(preds)
			print('\n Discriminator (0.5 optimum):' + str(d))
			discriminator[i]=d
			# Generate fake samples using the fake data generator function
			x_fake = fakedataGenerator(gen_model,batch,features)
			# Saving the plots
			for j in range(3*10):
				plt.subplot(3,10,j+1)
				# turn off axis 
				plt.axis('off')
				plt.imshow(x_fake[j,:,:,0],cmap='gray')                       
			filename = './data/MNIST_Digit_%i_Epoch_%03d.png' % (digit, i)
			plt.savefig(filename)
			plt.close()

	gan_model.save('./models/digits/gan_' + str(digit))
	gen_model.save('./models/digits/gen_' + str(digit))
	disc_model.save('./models/digits/disc_' + str(digit))
   
	return discriminator, gen_model


digits = [0,1]

for digit in digits:
	print('Starting digit ' + str(digit))
	r, gen_model = train(batch, features, nEpochs, reportEpochs, digit)
 
 
	mnist_synth = fakedataGenerator(gen_model,30,features)
	# Visualizing the plots
	for i in range(rows*cols):
   		plt.subplot(rows,cols,i+1)
   		# turn off axis 
   		plt.axis('off')
   		plt.imshow(mnist_synth[i,:,:,0],cmap='gray')
	plt.show()