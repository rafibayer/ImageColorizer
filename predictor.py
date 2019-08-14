from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
import numpy as np
import matplotlib.pyplot as plt


# Model reconstruction from JSON file
with open('model_arch.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model.h5')


import math

# load the l component of an image (LAB colorspace)
def load_lab_bw(path, size):
  image = img_to_array(load_img(path, target_size=size)) / 255
  lab_image = rgb2lab(image)
  lab_image = (lab_image + [0, 128, 128]) / [100, 255, 255] # normalize

  bw = lab_image[:,:,0]
  return bw


# splice a square bw image of im_size^2 into square of split_size^2
def splice_square_image(bw_image, im_size, split_size):
  dim = im_size / split_size
  if dim != int(dim):
    raise Exception("Error, please use a square number for tiles")
  
  dim = int(dim)
  # size of chunk
  C = split_size
  
  bw_chunks = []
  
  for x in range(dim):
    for y in range(dim):
      bw_chunks.append(bw_image[x*C : x*C + C, y*C : y*C + C])


  bw_chunks = np.expand_dims(np.array(bw_chunks), 3)
  return bw_chunks


# bw_chunks = splice_square_image(image, 2048, 256)
def predict_color(images):
    return model.predict(images)



# reconstruct an image of im_size^2 split into split_size^2 chunks, 
# use bw image for l component of LAB and color as color predictions
def reconstruct_img(im_size, split_size, bw, color):
  # calculate sidelength
  dim = (im_size / split_size)
  if dim != int(dim):
    raise Exception("Error, please use a square number for tiles, used: {}".format(dim))
  
  dim = int(dim)
  j = 0
  C = split_size

  result = np.zeros((im_size, im_size, 3))

  for x in range(dim):
    for y in range(dim):
      out = np.zeros((split_size, split_size, 3))

      out[:,:,0] = bw[j][:,:,0]
      out[:,:,1:] = color[j]

      out = (out * [100, 255, 255]) - [0, 128, 128]
      out = lab2rgb(out)

      result[x*C: x*C + C, y*C: y*C + C,:] = out
      j += 1
      
  return result

def colorizeImg(path, img_size):
    image = load_lab_bw(path, (img_size,img_size))
    bw_chunks = splice_square_image(image, img_size, 256)
    pred = predict_color(bw_chunks)
    return reconstruct_img(img_size, 256, bw_chunks, pred)
