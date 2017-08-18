
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to Image file
imgPath = "../test_dataset/IMG/sample.jpg"

# Read Image
image = cv2.imread(imgPath)

# Copy Images for each channel
npImageR = np.copy(image)
npImageG = np.copy(image)
npImageB = np.copy(image)

# Process to extract each channel (R,G,B) => img.shape = (r,c,3)
npImageR[:,:,[1,2]] = 0			# Extract Red channel
npImageG[:,:,[0,2]] = 0			# Extract Green channel
npImageB[:,:,[0,1]] = 0			# Extract Blue channel

# Extract each channel (R, G, B) => img.shape = (r,c) 
b, g, r = cv2.split(image)

# Extract each channel (R, G, B) => img.shape = (r,c) 
bChannel = image[:,:,0]
gChannel = image[:,:,1]
rChannel = image[:,:,2]

# Display/Plot Image
plt.subplot(1, 3, 1)
plt.imshow(npImageR)
plt.subplot(1, 3, 2)
plt.imshow(npImageG)
plt.subplot(1, 3, 3)
plt.imshow(npImageB)
plt.show()