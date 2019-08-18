from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfile
from PIL import ImageTk,Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from predictor import Predictor
import os

import matplotlib.pyplot as plt


IMAGE_SIZE = 512

pred = Predictor()

originalImage = None
colorImage = None

outputArray = None

root = Tk()

################## CONTROLS #########################

controls = Frame(root, height=600, width=300)
controls.grid(row=0, column=0, sticky=NW)

Label(controls, text="Options:").grid(row=0, column=0, sticky=W)

Label(controls, text="Target Resolution:").grid(row=1, column=0, sticky=W)
resolutionEntry = Entry(controls)
resolutionEntry.insert(END, "512")
resolutionEntry.grid(row=1, column=1, sticky=W)

chooseButton = Button(controls, text="Choose Image")
chooseButton.grid(row=2, column=0, sticky=W)

saveButton = Button(controls, text="Save Image")
saveButton.grid(row=2, column=1, sticky=W)

################## IMAGES #########################

images = Frame(root)
images.grid(row=0, column = 1, sticky=E) 

Label(images, text="Original").grid(row=0, column=0, sticky=W)
originalImage = ImageTk.PhotoImage(load_img("blank.jpg", target_size = (IMAGE_SIZE, IMAGE_SIZE)))
originalImageLabel = Label(images, image = originalImage)
originalImageLabel.grid(row=1, column = 0)

Label(images, text="Colorized").grid(row=0, column=1, sticky=W)
colorImage = ImageTk.PhotoImage(load_img("blank.jpg", target_size = (IMAGE_SIZE, IMAGE_SIZE)))
colorImageLabel = Label(images, image = colorImage)
colorImageLabel.grid(row = 1, column = 1)

################## BUTTONS & BINDINGS #########################


def updateOriginalImage():
    originalImageLabel.configure(image = originalImage)

def updateColorImage():
    colorImageLabel.configure(image = colorImage)

def predict_image(image_path, size):
    global originalImage
    global colorImage
    originalImage = ImageTk.PhotoImage(load_img(image_path, target_size = (IMAGE_SIZE, IMAGE_SIZE)))
    updateOriginalImage()

    result = pred.colorizeImg(image_path, size)
    global outputArray
    outputArray = result

    rescale = result * 255
    colorImage = ImageTk.PhotoImage(Image.fromarray(np.uint8(rescale)).resize((IMAGE_SIZE, IMAGE_SIZE)))
    updateColorImage()
    
def chooseImageButton(event):
    image_path = askopenfilename(title = "Select file", filetypes = [("jpeg files","*.jpg")])
    predict_image(image_path, getResolution())

def saveImageButton(event):
    f = asksaveasfile(mode='w', defaultextension=".jpg")
    if f is None:
        return
    
    im = Image.fromarray(np.uint8(outputArray*255))
    im.save(f)

def getResolution():
    return int(resolutionEntry.get())

chooseButton.bind("<Button-1>", chooseImageButton)
saveButton.bind("<Button-1>", saveImageButton)
root.mainloop()