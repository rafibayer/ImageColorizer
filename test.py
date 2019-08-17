from predictor import *
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image


path = "seattle.jpg" # image to colorize in images/ folder
out_name = "seattle_color.jpg" # output filename in output/ folder
img_size = 1024 # size to open the image as (try the nearest power of 2 that is > 256)

pred = Predictor()

image = pred.colorizeImg("images/{}".format(path), img_size)
matplotlib.image.imsave("output/{}".format(out_name), image)

pilImage = Image.fromarray((image * 255).astype("uint8"))
pilImage.show()

#plt.imshow(image)
#plt.show()

