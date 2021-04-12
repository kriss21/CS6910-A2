#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf


# In[2]:


from keras.layers import Conv2D,Dense,Activation,MaxPooling2D,Flatten,BatchNormalization,Dropout,InputLayer
from keras.models import Sequential


# In[3]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# In[4]:


class cnn_model:
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=input_shape))
        
    def add_cnn_block(self,nfilters=16,filter_size=(2,2),activation='relu',pool_size=(2,2),batch_norm='yes',dropout=None):
        self.model.add(Conv2D(nfilters,filter_size,padding="same"))
        if batch_norm == 'yes':
            self.model.add(BatchNormalization())
        self.model.add(Activation(activation))
        if dropout is not None:
            self.model.add(Dropout(dropout))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        
    def add_dense_layer(self,neurons=16):
        self.model.add(Flatten())
        self.model.add(Dense(neurons,activation='relu'))
        
    def build_model(self,output,loss='categorical_crossentropy',lr=1e-4):
        self.model.add(Dense(output,activation='softmax'))
        optz = keras.optimizers.Adam(lr)
        self.model.compile(optimizer=optz,loss=loss,metrics=['accuracy'])
#         self.model.summary()
        return self.model


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[6]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

pref = '../inaturalist_12K/'
 
train_generator = train_datagen.flow_from_directory(
    pref+'train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical')

val_generator = test_datagen.flow_from_directory(
    pref+'val',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical')


# In[7]:


classes = list(train_generator.class_indices.keys())
classes


# In[8]:


x,y = train_generator.next()

img = x[0]
plt.imshow(img)
plt.show()
print(classes[np.argmax(y[0])])


# In[9]:


def make_model(filters,filter_size,batch_norm,dropout,dense_size,lr):
    keras.backend.clear_session()
    model = cnn_model((224,224,3))
    for i in range(5):
        model.add_cnn_block(nfilters=filters[i],filter_size=filter_size[i],batch_norm=batch_norm,dropout=dropout)
    model.add_dense_layer(dense_size)
    training = model.build_model(10,lr=lr)
    return training




import wandb
from wandb.keras import WandbCallback


# In[11]:


sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [7]
        },
        'lr': {
            'values': [1e-5,1e-6]
        },
        'filters': {
            'values': [[32,64,128,256,512],
                       [64,64,64,64,64],
                       [512,256,128,64,32]
                      ]
        },
        'filter_size': {
            'values': [[(11,11),(7,7),(5,5),(3,3),(2,2)],
                       [(2,2),(3,3),(3,3),(5,5),(5,5)],
                       [(3,3),(3,3),(3,3),(3,3),(3,3)]
                      ]
        },
        'dense_size': {
            'values': [128,512,1024]
        },
        'batch_norm': {
            'values': ["yes","no"]
        },
        'dropout': {
            'values': [0.0,0.2,0.5]
        }
    }
}


# In[12]:


sweep_id = wandb.sweep(sweep_config,entity = "notarchana" , project = "cs6910-A1")


# In[13]:


# The sweep calls this function with each set of hyperparameters
def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'epochs': 7,
        'lr': 1e-5,
        'filters': [32,64,128,256,512],
        'filter_size': [(11,11),(7,7),(5,5),(3,3),(2,2)],
        'dense_size': 1024,
        'batch_norm': "yes",
        'dropout': None
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults,name="cs6910-a2")
    
    cfg = wandb.config
    
    name = f'flt_{cfg.filters}_fltsz_{cfg.filter_size}_bn_{cfg.batch_norm}_lr_{cfg.lr}_do_{cfg.dropout}_dsz_{cfg.dense_size}'
    wandb.run.name = name
    wandb.run.save()
    
    # Config is a variable that holds and saves hyperparameters and inputs
    
    model = make_model(cfg.filters,cfg.filter_size,cfg.batch_norm,cfg.dropout,cfg.dense_size,cfg.lr)
    print("model building done")
    trained = model.fit(train_generator,
                steps_per_epoch=500,
                epochs=cfg.epochs,
                validation_data=val_generator,
                validation_steps=100,
                callbacks=[WandbCallback(monitor='val_accuracy',mode='max')]
    )

    print("model training done")


# In[14]:


np.random.seed(42)
wandb.agent(sweep_id, train,count=20)

