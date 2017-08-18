
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform a warp perspective transformation
def perspectTransform(img, src, dst):
    w = img.shape[1]
    h = img.shape[0]

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped

# Path to Image file
imgPath = "../test_dataset/IMG/example_grid1.jpg"

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

# Draw Source and destination points on images (in blue) before plotting
cv2.polylines(image, np.int32([source]), True, (0, 0, 255), 3)
cv2.polylines(warpImage, np.int32([destination]), True, (0, 0, 255), 3)

# Display/Plot Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(warpImage, cmap = 'gray')
plt.show()