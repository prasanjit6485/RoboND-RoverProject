
import cv2
import numpy as np

# Function to perform a color threshold
def colorThresh(img, rgbThresh=(0, 0, 0)):
    binaryImage = np.zeros((img.shape[0],img.shape[1]),dtype = 'uint8')

    # Extract each channel
    b,g,r = cv2.split(img)
    
    # Apply the thresholds for RGB
    aboveThresh = (b > rgbThresh[0]) & (g > rgbThresh[1]) & (r > rgbThresh[1])
    binaryImage[aboveThresh] = 255

    return binaryImage

# Function to perform a warp perspective transformation
def perspectTransform(img, src, dst):
    w = img.shape[1]
    h = img.shape[0]

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped

# Function to perform a warp perspective transformation
def roverCoords(binaryImg):
    # Identify nonzero pixels
    ypos, xpos = binaryImg.nonzero()
    
    # Calculate pixel positions with reference to the rover position being at 
    # the center bottom of the image.  
    x_pixel = -(ypos - binaryImg.shape[0]).astype(np.float)
    y_pixel = -(xpos - binaryImg.shape[1]/2 ).astype(np.float)

    return x_pixel, y_pixel    