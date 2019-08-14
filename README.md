# ImageColorizer
Colorizing images using a convolutional neural network trained with google Colab


## Creating and training the model
The neural network for this project was created using keras and trained on google colab using an intel image dataset from kaggle

## Colorspace
My original idea for this project was to use a black and white image (N x N x 1) to predict the corresponding RBG image (N x N x 3).  
What I found is that the entire image would take on a fairly uniform color, even with different activations and loss functions.  
After doing some research I switched to LAB colorspace, this is still a 3-channel colorspace but the first channel, L, is just a scalar describing the brightness of each pixel.  
Due to this property, we can recover the L component from the original black and white image, and then we only have to predict the A and B components.  
This means the network doesn't have to spend as much time learning to reconstruct the features of the image, and instead only really has to predict A and B, the color components of each pixel.


## Variable image sizes
The underlying neural network only works on images of the shape (256 x 256 x 1), however since this is relatively small, it would severly limit the detail of colorized images if we just rescaled them all.  
To counteract this, I implemented a set of methods to splice the images into N squares of (256 x 256 x 1).  
The only downside of this method is that images must be square, and their dimensions must be a multiple of 256.  
In my tests, I have found that adding whitespace as padding and then cropping the image can help with images that don't have a 1:1 aspect ratio,  
and it is often possible to stretch or compress images to a near power of 2, for example (500x500) -> (512x512) with minimal detail loss.  
