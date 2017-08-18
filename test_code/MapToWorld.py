
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ExtraFunction import colorThresh, perspectTransform, roverCoords

# Define a function to apply a rotation to pixel positions
def rotatePix(xpix, ypix, yaw):
    # Convert yaw to radians
    yawRad = yaw * np.pi / 180
    xpixRotated = (xpix * np.cos(yawRad)) - (ypix * np.sin(yawRad))
                            
    ypixRotated = (xpix * np.sin(yawRad)) + (ypix * np.cos(yawRad))
    
    # Return the result  
    return xpixRotated, ypixRotated

# Define a function to perform a translation
def translatePix(xpixRot, ypixRot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpixTranslated = (xpixRot / scale) + xpos
    ypixTranslated = (ypixRot / scale) + ypos
    
    # Return the result  
    return xpixTranslated, ypixTranslated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pixToWorld(xpix, ypix, xpos, ypos, yaw, worldSize, scale):
    # Apply rotation
    xpixRot, ypixRot = rotatePix(xpix, ypix, yaw)

    # Apply translation
    xpixTran, ypixTran = translatePix(xpixRot, ypixRot, xpos, ypos, scale)
    
    # Clip to worldSize
    xPixWorld = np.clip(np.int_(xpixTran), 0, worldSize - 1)
    yPixWorld = np.clip(np.int_(ypixTran), 0, worldSize - 1)
    
    # Return the result
    return xPixWorld, yPixWorld

# Path to Image file
imgPath = "../test_dataset/IMG/sample.jpg"

# Read Image
image = cv2.imread(imgPath)

# Rover yaw values will come as floats from 0 to 360
# Generate a random value in this range
# Note: you need to convert this to radians
# before adding to pixel_angles
roverYaw = np.random.random(1)*360

# Generate a random rover position in world coords
# Position values will range from 20 to 180 to 
# avoid the edges in a 200 x 200 pixel world
roverXPos = np.random.random(1)*160 + 20
roverYPos = np.random.random(1)*160 + 20

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

# Generate 200 x 200 pixel worldmap
worldmap = np.zeros((200, 200))
scale = 10

# Get navigable pixel positions in world coords
xWorld, yWorld = pixToWorld(xpix, ypix, roverXPos, 
                                roverYPos, roverYaw, 
                                worldmap.shape[0], scale)
# Add pixel positions to worldmap
worldmap[yWorld, xWorld] += 1
print('Xpos =', roverXPos, 'Ypos =', roverYPos, 'Yaw =', roverYaw)

# Display/Plot Image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
f.tight_layout()
ax1.plot(xpix, ypix, '.')
ax1.set_title('Rover Space', fontsize=40)
ax1.set_ylim(-160, 160)
ax1.set_xlim(0, 160)
ax1.tick_params(labelsize=20)

ax2.imshow(worldmap, cmap='gray')
ax2.set_title('World Space', fontsize=40)
ax2.set_ylim(0, 200)
ax2.tick_params(labelsize=20)
ax2.set_xlim(0, 200)
plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
plt.show()