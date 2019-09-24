#!/usr/bin/env python
# coding: utf-8

# ## Color image demosaicing

#  The input to the algorithm is a single image *im*, an N x M array of numbers between 0.0 and 1.0. These are measurements in the format shown in Figure 2, i.e. the top left pixel *im(0, 0)* is red, *im(0, 1)* is green, *im(1, 0)* is also green and *im(1, 1)* refers to the blue channel.
#
# <img src="mosaic.png">

# First, run the following code to load the input images.

# In[ ]:


import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

images = ['balloons.jpg', 'candy.jpg', 'cat.jpg', 'ip.jpg',
          'puppy.jpg', 'squirrel.jpg', 'tree.jpg']

def show_image(img):
    plt.imshow(img, interpolation='nearest')


# In[ ]:


'''This array contains the mosaiced images which have to be demosaiced.'''
mosaiced_images = []
gtruth_images = []

'''A function to generate the mosaiced images from the given RGB images.'''
def generate_mosaic(image):
    mosaic = image[:, :, 1].copy() # green
    mosaic[::2, ::2] = image[::2, ::2, 0] # red
    mosaic[1::2, 1::2] = image[1::2, 1::2, 2] # blue
    return mosaic

'''A function to generate the mosaiced images for all the input images.'''
def generate_input(mosaiced_images, gtruth_images):
    for file in images:
        image = imread(file)
        image = image / 255.0
        gtruth_images.append(image)
        mosaiced_images.append(generate_mosaic(image))

generate_input(mosaiced_images, gtruth_images)


# ## Visualize the Images
#
# To gain intuition, let's visualize the image encoded in the Bayer pattern. Run the code below to see both the mosaic image and the original image. In this problem, you will take the image encoded in a Bayer pattern and recover the full color image.

# In[51]:


f, axarr = plt.subplots(1,2)
axarr[0].imshow(mosaiced_images[0], cmap='gray')
axarr[1].imshow(gtruth_images[0], cmap='gray')


# ## Implement your code here
#
# There are several algorithms that we can use to recover the original image. You will implement three of them:
#
# 1) **nearest neighbour** algorithm
#
# 2) **linear interpolation** algorithm.
#
# 3) **adaptive gradient-based** algorithm.
#
#

# In[ ]:


'''Methods to demosaic the image; the input is a mosaiced image and the output should be a
demosaiced image with R, G and B channels. The dimensions of the images should be the
same.'''

def demosaic_nn(im):
    #Initializine channel setups
    redChannel = np.zeros((im.shape[0],im.shape[1]))
    greenChannel = np.zeros((im.shape[0],im.shape[1]))
    blueChannel = np.zeros((im.shape[0],im.shape[1]))

    #Build the channels. All of the if conditions test the boundary conditions
    #in order to pull values from other cells.
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):

            if r%2 == 0 and c%2 == 0:
                redChannel[r,c] = im[r,c]
                greenChannel[r,c] = im[r,c+1] if c < im.shape[1] - 1 else im[r,c-1]
                if r < im.shape[0] - 1:
                    blueChannel[r,c] = im[r+1,c+1] if c < im.shape[1] - 1 else im[r+1,c-1]
                else:
                    blueChannel[r,c] = im[r-1,c+1] if c < im.shape[1] - 1 else im[r-1,c-1]

            elif r%2 == 0 and c%2 == 1:
                redChannel[r,c] = im[r,c+1] if c < im.shape[1] - 1 else im[r,c-1]
                greenChannel[r,c] = im[r,c]
                blueChannel[r,c] = im[r+1,c] if r < im.shape[0] - 1 else im[r-1,c]
            elif r%2 == 1 and c%2 == 0:
                redChannel[r,c] = im[r+1,c] if r < im.shape[0] - 1 else im[r-1,c]
                greenChannel[r,c] = im[r,c]
                blueChannel[r,c] = im[r,c+1] if c < im.shape[1] - 1 else im[r,c-1]

            elif r%2 == 1 and c%2 == 1:
                if r < im.shape[0] - 1:
                    redChannel[r,c] = im[r+1,c+1] if c < im.shape[1] - 1 else im[r+1,c-1]
                else:
                    redChannel[r,c] = im[r-1,c+1] if c < im.shape[1] - 1 else im[r-1,c-1]

                greenChannel[r,c] = im[r,c+1] if c < im.shape[1] - 1 else im[r,c-1]
                blueChannel[r,c] = im[r,c]




    return np.stack((redChannel,greenChannel,blueChannel),-1)

def demosaic_interpolate(im):
    #Initializine channel setups
    redChannel = np.zeros((im.shape[0],im.shape[1]))
    greenChannel = np.zeros((im.shape[0],im.shape[1]))
    blueChannel = np.zeros((im.shape[0],im.shape[1]))

    #Build the channels. Since there are more conditions than NN, I will first
    #do averaging off of the "inner" square - ignoring the boundaries

    for r in range(1,im.shape[0]-1):
        for c in range(1,im.shape[1]-1):

            if r%2 == 0 and c%2 == 0:
                redChannel[r,c] = im[r,c]
                greenChannel[r,c] = (im[r,c+1] + im[r,c-1] + im[r-1,c] + im[r+1,c])/4
                blueChannel[r,c] = (im[r+1,c+1] + im[r-1,c+1] + im[r-1,c-1] + im[r+1,c-1])/4

            elif r%2 == 0 and c%2 == 1:
                redChannel[r,c] = (im[r,c+1] + im[r,c-1])/2
                greenChannel[r,c] = im[r,c]
                blueChannel[r,c] = (im[r+1,c] + im[r-1,c])/2

            elif r%2 == 1 and c%2 == 0:
                redChannel[r,c] = (im[r+1,c] + im[r-1,c])/2
                greenChannel[r,c] = im[r,c]
                blueChannel[r,c] = (im[r,c+1] + im[r,c-1])/2

            elif r%2 == 1 and c%2 == 1:
                redChannel[r,c] = (im[r+1,c+1] + im[r-1,c+1] + im[r-1,c-1] + im[r+1,c-1])/4
                greenChannel[r,c] = (im[r,c+1] + im[r,c-1] + im[r-1,c] + im[r+1,c])/4
                blueChannel[r,c] = im[r,c]

    #Now we find the averages for the edges. Corners will be last

    #Top and Bottom
    b = im.shape[0]-1
    for c in range(1,im.shape[1]-1):

        if c%2 == 0:

            redChannel[0,c] = im[0,c]
            redChannel[b,c] = im[b,c] if b%2==0 else im[b-1,c]
            greenChannel[0,c] = (im[0,c-1] + im[0,c+1])/2
            greenChannel[b,c] = (im[b,c-1] + im[b,c+1])/2 if b%2==0 else im[b,c]
            blueChannel[0,c] = (im[1,c-1] + im[1,c+1])/2
            blueChannel[b,c] = (im[b-1,c-1] + im[b-1,c+1])/2 if b%2==0 else (im[b,c-1] + im[b,c+1])/2

        else:
            redChannel[0,c] = (im[0,c-1] + im[0,c+1])/ 2
            if b%2 == 0:
                redChannel[b,c] = (im[b,c-1]+im[b,c+1]) /2
            else:
                redChannel[b,c] = (im[b-1,c-1] + im[b-1,c+1])/2

            greenChannel[0,c] = im[0,c]
            greenChannel[b,c] = im[b,c] if b%2==0 else (im[b,c-1]+im[b,c+1])/2

            blueChannel[0,c] = im[1,c]
            blueChannel[b,c] = im[b-1,c] if b%2==0 else im[b,c]

    #Left and Right
    r = im.shape[1]-1

    for x in range(1,b):

        if x%2 == 0:

            redChannel[x,0] = im[x,0]
            redChannel[x,r] = im[x,r] if r%2==0 else im[x,r-1]
            greenChannel[x,0] = (im[x-1,0] + im[x+1,0])/2
            greenChannel[x,r] = (im[x-1,r] + im[x+1,r])/2 if r%2==0 else im[x,r]
            blueChannel[x,0] = (im[x-1,1] + im[x+1,1])/2
            blueChannel[x,r] = (im[x-1,r-1] + im[x+1,r-1])/2 if r%2==0 else (im[x-1,r] + im[x+1,r])/2

        else:
            redChannel[x,0] = (im[x-1,0] + im[x+1,0])/ 2
            if r%2 == 0:
                redChannel[x,r] = (im[x-1,r]+im[x+1,r]) /2
            else:
                redChannel[x,r] = (im[x-1,r-1] + im[x+1,r-1])/2

            greenChannel[x,0] = im[x,0]
            greenChannel[x,r] = im[x,r] if r%2==0 else (im[x-1,r]+im[x+1,r])/2

            blueChannel[0,r] = im[1,r]
            blueChannel[x,r] = im[x,r-1] if r%2==0 else im[x,r]







    #Now the corners
    redChannel[0,0] = im[0,0]
    greenChannel[0,0] = (im[0,1]+im[1,0])/2
    blueChannel[0,0] = im[1,1]

    redChannel[0,r] = im[0,r] if r%2==0 else im[0,r-1]
    greenChannel[0,r] = (im[0,r-1]+im[1,r])/2 if r%2==0 else im[0,r]
    blueChannel[0,r] = im[1,r-1] if r%2==0 else im[1,r]

    redChannel[b,0] = im[b,0] if r%2==0 else im[b-1,0]
    greenChannel[b,0] = (im[b-1,0]+im[b,1])/2 if b%2==0 else im[b,0]
    blueChannel[b,0] = im[b-1,1] if b%2==0 else im[b,1]

    if b%2 == 0:
        redChannel[b,r] = im[b,r] if r%2==0 else im[b,r-1]
        greenChannel[b,r] = (im[b,r-1]+im[b-1,r])/2 if r%2==0 else im[b,r]
        blueChannel[b,r] = im[b-1,r-1] if r%2==0 else im[b-1,r]
    else:
        redChannel[b,r] = im[b-1,r] if r%2==0 else im[b-1,r-1]
        greenChannel[b,r] = (im[b,r-1]+im[b-1,r])/2 if r%2!=0 else im[b,r]
        blueChannel[b,r] = im[b,r-1] if r%2==0 else im[b,r]

    return np.stack((redChannel,greenChannel,blueChannel),-1)

def demosaic_gradient(im):
    '''
    According to our model, the edges and green cells will follow an ordinary
    linear interpolation. Therefore, some code snippets I will recopy. There
    is a lot of similarity.

    We will just now assign the smallest gradient average pair to interpolate.
    Therefore, due to the need of 2-pair colors, we can only run this algo
    on the "inner" square, making implimentation much easier since we don't have
    to worry about edge cases.
    '''

    redChannel = np.zeros((im.shape[0],im.shape[1]))
    greenChannel = np.zeros((im.shape[0],im.shape[1]))
    blueChannel = np.zeros((im.shape[0],im.shape[1]))


    for r in range(1,im.shape[0]-1):
        for c in range(1,im.shape[1]-1):

            if r%2 == 0 and c%2 == 0:
                redChannel[r,c] = im[r,c]

                if abs(im[r,c-1] - im[r,c+1]) > abs(im[r-1,c] - im[r+1,c]):
                    greenChannel[r,c] = (im[r-1,c] + im[r+1,c])/2
                else:
                    greenChannel[r,c] = (im[r,c-1] + im[r,c+1])/2

                if abs(im[r-1,c-1] - im[r+1,c+1]) > abs(im[r-1,c+1]-im[r+1,c-1]):
                    blueChannel[r,c] = (im[r-1,c+1]+im[r+1,c-1])/2
                else:
                    blueChannel[r,c] = (im[r-1,c-1] + im[r+1,c+1])/2

            elif r%2 == 0 and c%2 == 1: #Stays the same; a green cell
                redChannel[r,c] = (im[r,c+1] + im[r,c-1])/2
                greenChannel[r,c] = im[r,c]
                blueChannel[r,c] = (im[r+1,c] + im[r-1,c])/2

            elif r%2 == 1 and c%2 == 0: #Stays the same; a green cell
                redChannel[r,c] = (im[r+1,c] + im[r-1,c])/2
                greenChannel[r,c] = im[r,c]
                blueChannel[r,c] = (im[r,c+1] + im[r,c-1])/2

            elif r%2 == 1 and c%2 == 1:

                if abs(im[r-1,c-1] - im[r+1,c+1]) > abs(im[r-1,c+1]-im[r+1,c-1]):
                    redChannel[r,c] = (im[r-1,c+1]+im[r+1,c-1])/2
                else:
                    redChannel[r,c] = (im[r-1,c-1] + im[r+1,c+1])/2

                if abs(im[r,c-1] - im[r,c+1]) > abs(im[r-1,c] - im[r+1,c]):
                    greenChannel[r,c] = (im[r-1,c] + im[r+1,c])/2
                else:
                    greenChannel[r,c] = (im[r,c-1] + im[r,c+1])/2

                blueChannel[r,c] = im[r,c]


    #Now we find the averages for the edges. Corners will be last

    #Top and Bottom
    b = im.shape[0]-1
    for c in range(1,im.shape[1]-1):

        if c%2 == 0:

            redChannel[0,c] = im[0,c]
            redChannel[b,c] = im[b,c] if b%2==0 else im[b-1,c]
            greenChannel[0,c] = (im[0,c-1] + im[0,c+1])/2
            greenChannel[b,c] = (im[b,c-1] + im[b,c+1])/2 if b%2==0 else im[b,c]
            blueChannel[0,c] = (im[1,c-1] + im[1,c+1])/2
            blueChannel[b,c] = (im[b-1,c-1] + im[b-1,c+1])/2 if b%2==0 else (im[b,c-1] + im[b,c+1])/2

        else:
            redChannel[0,c] = (im[0,c-1] + im[0,c+1])/ 2
            if b%2 == 0:
                redChannel[b,c] = (im[b,c-1]+im[b,c+1]) /2
            else:
                redChannel[b,c] = (im[b-1,c-1] + im[b-1,c+1])/2

            greenChannel[0,c] = im[0,c]
            greenChannel[b,c] = im[b,c] if b%2==0 else (im[b,c-1]+im[b,c+1])/2

            blueChannel[0,c] = im[1,c]
            blueChannel[b,c] = im[b-1,c] if b%2==0 else im[b,c]

    #Left and Right
    r = im.shape[1]-1

    for x in range(1,b):

        if x%2 == 0:

            redChannel[x,0] = im[x,0]
            redChannel[x,r] = im[x,r] if r%2==0 else im[x,r-1]
            greenChannel[x,0] = (im[x-1,0] + im[x+1,0])/2
            greenChannel[x,r] = (im[x-1,r] + im[x+1,r])/2 if r%2==0 else im[x,r]
            blueChannel[x,0] = (im[x-1,1] + im[x+1,1])/2
            blueChannel[x,r] = (im[x-1,r-1] + im[x+1,r-1])/2 if r%2==0 else (im[x-1,r] + im[x+1,r])/2

        else:
            redChannel[x,0] = (im[x-1,0] + im[x+1,0])/ 2
            if r%2 == 0:
                redChannel[x,r] = (im[x-1,r]+im[x+1,r]) /2
            else:
                redChannel[x,r] = (im[x-1,r-1] + im[x+1,r-1])/2

            greenChannel[x,0] = im[x,0]
            greenChannel[x,r] = im[x,r] if r%2==0 else (im[x-1,r]+im[x+1,r])/2

            blueChannel[0,r] = im[1,r]
            blueChannel[x,r] = im[x,r-1] if r%2==0 else im[x,r]







    #Now the corners
    redChannel[0,0] = im[0,0]
    greenChannel[0,0] = (im[0,1]+im[1,0])/2
    blueChannel[0,0] = im[1,1]

    redChannel[0,r] = im[0,r] if r%2==0 else im[0,r-1]
    greenChannel[0,r] = (im[0,r-1]+im[1,r])/2 if r%2==0 else im[0,r]
    blueChannel[0,r] = im[1,r-1] if r%2==0 else im[1,r]

    redChannel[b,0] = im[b,0] if r%2==0 else im[b-1,0]
    greenChannel[b,0] = (im[b-1,0]+im[b,1])/2 if b%2==0 else im[b,0]
    blueChannel[b,0] = im[b-1,1] if b%2==0 else im[b,1]

    if b%2 == 0:
        redChannel[b,r] = im[b,r] if r%2==0 else im[b,r-1]
        greenChannel[b,r] = (im[b,r-1]+im[b-1,r])/2 if r%2==0 else im[b,r]
        blueChannel[b,r] = im[b-1,r-1] if r%2==0 else im[b-1,r]
    else:
        redChannel[b,r] = im[b-1,r] if r%2==0 else im[b-1,r-1]
        greenChannel[b,r] = (im[b,r-1]+im[b-1,r])/2 if r%2!=0 else im[b,r]
        blueChannel[b,r] = im[b,r-1] if r%2==0 else im[b,r]

    return np.stack((redChannel,greenChannel,blueChannel),-1)



    return np.stack((redChannel,greenChannel,blueChannel),-1)


'''These lists will contain the output images of each of the demosaicing methods.'''
nn_output = []
interpolation_output = []
gradient_output = []

def demosaic_image(image, method='NN'):
    if method == 'NN':
        return demosaic_nn(image)
    elif method == 'interpolate':
        return demosaic_interpolate(image)
    elif method == 'gradient':
        return demosaic_gradient(image)
    else:
        raise ValueError('Unknown method')

def generate_output(mosaiced_images):

    # Demosaicing using the nearest neighbour method
    for image in mosaiced_images:
        op = demosaic_image(image, method='NN')
        nn_output.append(op)

    # Demosaicing using the linear interpolation method
    for image in mosaiced_images:
        op = demosaic_image(image, method='interpolate')
        interpolation_output.append(op)

    # Demosaicing using the gradient-based method
    for image in mosaiced_images:
        op = demosaic_image(image, method='gradient')
        gradient_output.append(op)

generate_output(mosaiced_images)

'''Modify the index of the image to show as an example here.'''
image_to_show = 2

plt.figure()
show_image(nn_output[image_to_show])
plt.figure()
show_image(interpolation_output[image_to_show])
plt.figure()
show_image(gradient_output[image_to_show])


# ## Evaluation



def calculate_mean_difference(output_images):
    sum = 0.0
    for idx, image in enumerate(output_images):
        pixel_error = np.absolute(image - gtruth_images[idx])
        sum += np.mean(pixel_error)
    sum /= len(output_images)

    return sum

def evaluate(threshold):
    error = calculate_mean_difference(nn_output)
    if  error < threshold:
        print("NN -> Likely passed! Mean error is", error)
    else:
        print("NN -> Failed! Mean error is", error)

    error = calculate_mean_difference(interpolation_output)
    if  error < threshold:
        print("Interpolation -> Likely passed! Mean error is", error)
    else:
        print("Interpolation -> Failed! Mean error is", error)

    error = calculate_mean_difference(gradient_output)
    if  error < threshold:
        print("Gradient -> Likely passed! Mean error is", error)
    else:
        print("Gradient -> Failed! Mean error is", error)

evaluate(threshold=0.03)
