import predictor
import matplotlib.pyplot as plt
import matplotlib


path = "seattle.jpg"
out_name = "seattle_color.jpg"
img_size = 1024

image = predictor.colorizeImg("images/{}".format(path), img_size)
matplotlib.image.imsave("output/{}".format(out_name), image)


plt.imshow(image)
plt.show()

