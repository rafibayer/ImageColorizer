import predictor
import matplotlib.pyplot as plt
import matplotlib



filename = "dude_color.jpg"
image = predictor.colorizeImg("dude.jpg", 1024)
matplotlib.image.imsave("output/{}".format(filename), image)


plt.imshow(image)
plt.show()

