from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

class Predictor:
  
  def __init__(self):
    # Model reconstruction from JSON file
    with open('model_arch.json', 'r') as f:
        self.model = model_from_json(f.read())

    # Load weights into the model
    self.model.load_weights('model.h5')



  # load the l component of an image (LAB colorspace)
  def load_lab_bw(self, path, size):
    image = img_to_array(load_img(path, target_size=size)) / 255 #open image as array and normalize values [0, 1]
    lab_image = rgb2lab(image)
    lab_image = (lab_image + [0, 128, 128]) / [100, 255, 255] # normalize LAB

    bw = lab_image[:,:,0] # Take just the L component, black and white image
    return bw


  # splice a square bw image of im_size^2 into square of split_size^2
  def splice_square_image(self, bw_image, im_size, split_size):
    dim = im_size / split_size # calculate the number of chunks for width and height
    if dim != int(dim):
      raise Exception("Error, please use a square number for tiles")
    
    dim = int(dim)
    # size of chunk
    C = split_size
    
    bw_chunks = [] # holds each chunk/tile 
    
    for x in range(dim):
      for y in range(dim):
        bw_chunks.append(bw_image[x*C : x*C + C, y*C : y*C + C]) # slice the image into chunks and append onto list


    bw_chunks = np.expand_dims(np.array(bw_chunks), 3) # add an extra dimension to satisfy model input shape
    return bw_chunks

  # predicts the color in a set of images, use to predict color for each chunk
  def predict_color(self, images):
      return self.model.predict(images)



  # reconstruct an image of im_size^2 split into split_size^2 chunks, 
  # use bw image for l component of LAB and color as color predictions
  def reconstruct_img(self, im_size, split_size, bw, color):
    # calculate sidelength
    dim = (im_size / split_size)
    if dim != int(dim):
      raise Exception("Error, please use a square number for tiles, used: {}".format(dim))
    
    dim = int(dim)
    j = 0 # chunk index variable
    C = split_size 

    result = np.zeros((im_size, im_size, 3)) # reconstructed output image

    for x in range(dim):
      for y in range(dim):
        out = np.zeros((split_size, split_size, 3))

        out[:,:,0] = bw[j][:,:,0] # L component from original black and white image
        out[:,:,1:] = color[j] # A and B components predicted by neural network

        out = (out * [100, 255, 255]) - [0, 128, 128] # denormalizing
        out = lab2rgb(out) # converting chunk back to RGB

        result[x*C: x*C + C, y*C: y*C + C,:] = out # placing reconstructed chunk into the final image
        j += 1
        
    return result

  # single function that takes in an image path and size, and returns a colorized image
  def colorizeImg(self, path, original_res):

      image = self.load_lab_bw(path, (original_res[0], original_res[1]))


      nearest = self.nearestSquareResolution(original_res)
      square_img = Image.new("L", (nearest, nearest), "white")
      image = Image.fromarray(np.uint8(image * 255), "L")

      square_img.paste(image)
      square_img = np.asarray(square_img) / 255.
      bw_chunks = self.splice_square_image(square_img, nearest, 256)
      pred = self.predict_color(bw_chunks)
      reconstructed = self.reconstruct_img(nearest, 256, bw_chunks, pred)
      crop = reconstructed[:original_res[0], :original_res[1]]


      return crop

  def nextPowOf2(self, n):
      return int(math.pow(2, math.ceil(math.log(n, 2))))


  def nearestSquareResolution(self, shape):
      return self.nextPowOf2(max(shape[0], shape[1]))
