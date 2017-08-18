
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to perform a color threshold
def colorThresh(img, rgbThresh=(0, 0, 0)):
    binaryImage = np.zeros((img.shape[0],img.shape[1]),dtype = 'uint8')

    # Extract each channel
    b,g,r = cv2.split(img)
    
    # Apply the thresholds for RGB
    aboveThresh = (b > rgbThresh[0]) & (g > rgbThresh[1]) & (r > rgbThresh[1])
    binaryImage[aboveThresh] = 255

    return binaryImage

def color_detection(img, lower = (170, 120, 0), upper = (230, 180, 60)):
	# create NumPy arrays from the lower, upper
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply the mask
	mask = cv2.inRange(img, lower, upper)
	masked_img = cv2.bitwise_and(img, img, mask = mask)

	# convert masked image to binary image
	gray_masked_img = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
	ret,threshold_img = cv2.threshold(gray_masked_img,1,255,cv2.THRESH_BINARY)

	return threshold_img

# Path to Image file
imgPath = "../test_dataset/test_data/sample.jpg"

# Read Image
# image = cv2.imread(imgPath)
image = mpimg.imread(imgPath)

#Define color selection criteria
redT = 160
greenT = 160
blueT = 160
rgbThreshold = (redT, greenT, blueT)

# Invoke color threshold function
# binImage = colorThresh(image,rgbThreshold)

binImage = color_detection(image)

y = np.mean(binImage)
print(y)

# Display/Plot Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(binImage, cmap = 'gray')
plt.show()