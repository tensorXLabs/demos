import streamlit as st
import numpy as np
import keras
import tensorflow as tf
from keras.layers import *
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image
from zipfile import ZipFile

st.title('Diffusion generator')

mod = keras.models.load_model('gen.h5')

n_steps = 500
beta = np.linspace(0.0001,0.04,n_steps)
alpha = 1-beta
alpha_bar = np.cumprod(alpha, axis = 0)

def p_xt1(xt, noise, t):
  alpha_t = alpha[t]
  alpha_bar_t = alpha_bar[t]
  eps_coef = ((1 - alpha_t) / (1 - alpha_bar_t) ** .5)[:,np.newaxis, np.newaxis, np.newaxis]
  mean = 1 / (alpha_t ** 0.5)[:,np.newaxis, np.newaxis, np.newaxis] * (xt - eps_coef * noise) # Note minus sign
  var = beta[t][:,np.newaxis, np.newaxis, np.newaxis]
  # eps = torch.randn(xt.shape, device=xt.device)
  eps = np.random.normal(0,1,xt.shape)
  return mean + (var ** 0.5) * eps 

# width = st.number_input('Insert patch dimension for side')

n_steps = 500
# number = st.number_input('Insert a patchsize')

number = st.selectbox('Select patch size',(4,16,32,64,128,256,512))

size = int(number)

import time

my_bar = st.progress(0)

if st.button('generate'):
  st.write('Now generating, please wait a few moments')
  x = np.random.normal(0,1,(5,size,size,3)) # Start with random noise
  ims = []
  ims.append(x)
  for i in range(n_steps):
    t = n_steps-i-1
    t = np.asarray([t for i in range(5)])
    pred_noise = mod.predict((x,t/n_steps))
    # print(pred_noise.shape)
    x = p_xt1(x, pred_noise, t)
    my_bar.progress(int(i/n_steps*100) + 1)
    if i % 99 ==0:
      print(i)
      ims.append(x)
    if i == 499:
      d = np.zeros((size,size*len(ims),3))
      j = 0
      for i in range(0,size*len(ims),size):
        d[:,i:i+size,:] = ims[j][0]
        d[:,i:i+1,:] = 0
        j+=1

      d = d* 255
      d = d.astype(np.uint8)
      img = Image.fromarray(d)

      st.image(img, caption='generated sequence', width=5*200)
###########################################################
      d = np.zeros((size,size*5,3))
      j = 0
      for i in range(0,size*5,size):
        d[:,i:i+size,:] = x[j]
        d[:,i:i+1,:] = 1
        j+=1

      d = d* 255
      d = d.astype(np.uint8)
      img = Image.fromarray(d)

      st.image(img, caption='generated ssamples', width=5*200)

      break


