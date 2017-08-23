
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

# Morphological operation on Binary/Grayscale image
def morphological_operation(image, operation, num_of_iterations = 1):
  # Check if image is grayscale/binary else return 0
  if len(image.shape) == 3:
    return 0

  # Define kernel [2x2] for morphological operation
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

  # Perform morphological operation
  if operation == 'erosion':
    morph_image = cv2.erode(image,kernel,iterations = num_of_iterations)
  elif operation == 'dilation':
    morph_image = cv2.dilate(image,kernel,iterations = num_of_iterations)
  elif operation == 'opening':
    morph_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = num_of_iterations)
  elif operation == 'closing':
    morph_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = num_of_iterations)

  return morph_image

# Function to perform a warp perspective transformation
def perspectTransform(img, src, dst):
    w = img.shape[1]
    h = img.shape[0]

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped

# Define a function to convert from image coords to rover coords
def rover_coords(binaryImg):
    # Identify nonzero pixels
    ypos, xpos = binaryImg.nonzero()
    
    # Calculate pixel positions with reference to the rover position being at 
    # the center bottom of the image.  
    x_pixel = -(ypos - binaryImg.shape[0]).astype(np.float)
    y_pixel = -(xpos - binaryImg.shape[1]/2 ).astype(np.float)

    return x_pixel, y_pixel 

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

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

for num in range(1,170):
  print("\n\nDataset:%s" %num)
  # Path to Image file
  # imgPath = "../test_dataset/rock_dataset/IMG2/rocksample%s.jpg" % num
  imgPath = "../test_dataset/obstacle_dataset/IMG/obstaclesample%s.jpg" % num
  # imgPath = "../test_dataset/obstaclesample.jpg"

  # Read Image
  # image = cv2.imread(imgPath)
  image = mpimg.imread(imgPath)

  # Define source and destination points for perspective transform
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

  # Apply perspective transform
  warpImage = perspectTransform(image,source,destination)

  #Define color selection criteria
  redT = 160
  greenT = 160
  blueT = 160
  rgbThreshold = (redT, greenT, blueT)

  # Invoke color threshold function
  binImage = colorThresh(warpImage,rgbThreshold)

  # binImage = color_detection(warpImage)

  # Apply morphological operation
  morphedImage = morphological_operation(binImage, 'dilation',6)

  y = np.mean(morphedImage)
  # print(y)

  # Convert map image pixel values to rover-centric coords
  xpix, ypix = rover_coords(morphedImage)  # Convert to rover-centric coords

  distances, angles = to_polar_coords(xpix, ypix) # Convert to polar coords
  avg_angle = np.mean(angles) # Compute the average angle

  # print(len(distances))
  # print((angles))
  # print(np.mean(distances))
  # print(avg_angle)

  negavg = angles[angles < 0]
  posavg = angles[angles > 0]

  print(len(angles))
  print(len(negavg))
  print(len(posavg))

# # Display/Plot Image
# plt.subplot(1, 4, 1)
# plt.imshow(image)
# plt.subplot(1, 4, 2)
# plt.imshow(warpImage)
# plt.subplot(1, 4, 3)
# plt.imshow(morphedImage, cmap = 'gray')
# plt.subplot(1, 4, 4)
# plt.plot(xpix, ypix, '.')
# plt.ylim(-160, 160)
# plt.xlim(0, 160)
# plt.show()

# # Rover yaw values will come as floats from 0 to 360
# # Generate a random value in this range
# # Note: you need to convert this to radians
# # before adding to pixel_angles
# roverYaw = np.random.random(1)*360

# # Generate a random rover position in world coords
# # Position values will range from 20 to 180 to 
# # avoid the edges in a 200 x 200 pixel world
# roverXPos = np.random.random(1)*160 + 20
# roverYPos = np.random.random(1)*160 + 20

# # Generate 200 x 200 pixel worldmap
# worldmap = np.zeros((200, 200))
# scale = 10

# # Get navigable pixel positions in world coords
# xWorld, yWorld = pixToWorld(xpix, ypix, roverXPos, 
#                                 roverYPos, roverYaw, 
#                                 worldmap.shape[0], scale)
# # Add pixel positions to worldmap
# worldmap[yWorld, xWorld] += 1
# print('Xpos =', roverXPos, 'Ypos =', roverYPos, 'Yaw =', roverYaw)

# # Display/Plot Image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
# f.tight_layout()
# ax1.plot(xpix, ypix, '.')
# ax1.set_title('Rover Space', fontsize=40)
# ax1.set_ylim(-160, 160)
# ax1.set_xlim(0, 160)
# ax1.tick_params(labelsize=20)

# ax2.imshow(worldmap, cmap='gray')
# ax2.set_title('World Space', fontsize=40)
# ax2.set_ylim(0, 200)
# ax2.tick_params(labelsize=20)
# ax2.set_xlim(0, 200)
# plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
# plt.show()

