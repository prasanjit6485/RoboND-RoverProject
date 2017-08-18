
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ExtraFunction import colorThresh, perspectTransform

# Function to perform a warp perspective transformation
def roverCoords(binaryImg):
    # Identify nonzero pixels
    ypos, xpos = binaryImg.nonzero()
    
    # Calculate pixel positions with reference to the rover position being at 
    # the center bottom of the image.  
    x_pixel = -(ypos - binaryImg.shape[0]).astype(np.float)
    y_pixel = -(xpos - binaryImg.shape[1]/2 ).astype(np.float)

    return x_pixel, y_pixel 

# Path to Image file
imgPath = "../test_dataset/IMG/sample.jpg"

# Read Image
image = cv2.imread(imgPath)

# Define a box in source (original) and destination (desired) coordinates
dstSize = 5 
bottomOffset = 6
width = image.shape[1]
height = image.shape[0]
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[width/2 - dstSize, height - bottomOffset],
                  [width/2 + dstSize, height - bottomOffset],
                  [width/2 + dstSize, height - 2*dstSize - bottomOffset], 
                  [width/2 - dstSize, height - 2*dstSize - bottomOffset],
                  ])

# Invoke color threshold function
warpImage = perspectTransform(image,source,destination)

#Define color selection criteria
redT = 160
greenT = 160
blueT = 160
rgbThreshold = (redT, greenT, blueT)

# Invoke color threshold function
binImage = colorThresh(warpImage,rgbThreshold)

# Extract x and y positions of navigable terrain pixels
# and convert to rover coordinates
xpix, ypix = roverCoords(binImage)

# Display/Plot Image
fig = plt.figure(figsize=(5, 7.5))
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
plt.show()