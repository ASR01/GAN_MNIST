from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten,Activation
from tensorflow.keras.layers import LeakyReLU,BatchNormalization
from tensorflow.keras.layers import Conv2D, UpSampling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam


# Generator model
def genModel(input_features):
    # Generator model
    Genmodel = Sequential()
    Genmodel.add(Dense(512,input_dim=input_features))
    Genmodel.add(Activation('relu'))
    Genmodel.add(BatchNormalization())

    Genmodel.add(Dense(7*7*64))
    Genmodel.add(Activation('relu'))
    Genmodel.add(BatchNormalization())
    # 14x14
    Genmodel.add(Reshape((7,7,64)))
    Genmodel.add(Conv2DTranspose(32,kernel_size=(5,5),strides=(2,2),padding='same'))
    Genmodel.add(Activation('relu'))
    Genmodel.add(BatchNormalization(axis = -1))
    # 28x28
    Genmodel.add(Conv2DTranspose(1,kernel_size=(5,5),strides=(2,2),padding='same'))
    Genmodel.add(Activation('tanh'))    
    return Genmodel


# Descriminator model

def discModel():
    Discmodel = Sequential()
    Discmodel.add(Conv2D(32,kernel_size=(5,5),strides=(2,2),padding='same',input_shape=(28,28,1)))    
    Discmodel.add(LeakyReLU(0.2))
    # second layer of convolutions
    Discmodel.add(Conv2D(64, kernel_size=(5,5), strides=(2, 2), padding='same'))   
    Discmodel.add(LeakyReLU(0.2))
    # Fully connected layers
    Discmodel.add(Flatten())
    Discmodel.add(Dense(512))
    Discmodel.add(LeakyReLU(0.2))
    Discmodel.add(Dense(1, activation='sigmoid'))
    Discmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005, beta_1 = 0.9,),metrics=['accuracy'])   
    return Discmodel 

# GAN Model 
def ganModel(gen_model,disc_model):
    # Discriminator model cannot be trained
    disc_model.trainable = False
    gan_model = Sequential()
    # First adding the generator model
    gan_model.add(gen_model)
    # Next adding the discriminator model without training the parameters
    gan_model.add(disc_model)
    # Compile the model for loss to optimise the Generator model
    gan_model.compile(loss='binary_crossentropy',optimizer = Adam(learning_rate=0.0005, beta_1 = 0.9,))    
    return gan_model