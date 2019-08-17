from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from predictor import Predictor

IMAGE_SIZE = 500

pred = Predictor()

#original_image = ImageTk.PhotoImage(load_img("blank.jpg", target_size = (IMAGE_SIZE, IMAGE_SIZE)))
original_image = ImageTk.PhotoImage(Image.open("blank.jpg"))
color_image = ImageTk.PhotoImage(Image.open("blank.jpg"))

def colorizeImage(path, resolution):

    print("colorizing...")
    color = pred.colorizeImg(path, resolution)

    original_image = ImageTk.PhotoImage(load_img(path, target_size = (IMAGE_SIZE, IMAGE_SIZE)))
    canvasOrig.image = original_image 

    color_image = ImageTk.PhotoImage(Image.fromarray((color * 255).astype("uint8")).resize((IMAGE_SIZE, IMAGE_SIZE)))
    canvasCol.image = color_image

    print("done!")


root = Tk()

controls = Frame(root, height=600, width=300)
controls.grid(row=0, column=0, sticky=NW)

Label(controls, text="Options:").grid(row=0, column=0, sticky=W)

Label(controls, text="Target Resolution:").grid(row=1, column=0, sticky=W)
resolutionEntry = Entry(controls)
resolutionEntry.grid(row=1, column=1, sticky=W)

chooseButton = Button(controls, text="Choose Image")
chooseButton.grid(row=2, column=0, sticky=W)

saveButton = Button(controls, text="Save Image")
saveButton.grid(row=2, column=1, sticky=W)

###############################################

images = Frame(root)
images.grid(row=0, column = 1, sticky=E) 

Label(images, text="Original").grid(row=0, column=0, sticky=W)
canvasOrig = Label(images, height = IMAGE_SIZE, width = IMAGE_SIZE)
canvasOrig.grid(row=1, column=0)
canvasOrig.image = original_image 



Label(images, text="Colorized").grid(row=0, column=1, sticky=W)
canvasCol = Label(images, height = IMAGE_SIZE, width = IMAGE_SIZE)
canvasCol.grid(row=1, column=1)
canvasCol.image = color_image 


#colorizeImage("images/nyc.jpg", 512)
  

root.mainloop()