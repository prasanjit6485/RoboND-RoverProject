import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 255
    # Return the binary image
    return color_select

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

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
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

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    image = Rover.img

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
    warpImage, mask = perspect_transform(image,source,destination)

    # Apply color threshold to identify navigable terrain
    nav_bin_image = color_thresh(warpImage)

    # Invert navigable terrain to identify obstacle terrain and also consider
    # the improvise field of view from Rover camera
    # obstacle_bin_image = 255 - nav_bin_image
    obstacle_bin_image = np.absolute(np.float32(nav_bin_image-255)*mask) * 255

    # Apply color threshold to identify rock samples
    rock_bin_image = color_detection(warpImage)

    # Apply morphological operation
    morphed_image = morphological_operation(rock_bin_image, 'dilation',6)

    Rover.vision_image[:,:,0] = obstacle_bin_image
    Rover.vision_image[:,:,1] = rock_bin_image
    Rover.vision_image[:,:,2] = nav_bin_image

    if(np.mean(morphed_image) > 0.1):
        Rover.rock_detected = True

        # Convert map image pixel values to rover-centric coords
        xpix, ypix = rover_coords(morphed_image)

        # Convert rover-centric pixel values to world coordinates
        scale = 10
        xWorld, yWorld = pix_to_world(xpix, ypix, Rover.pos[0], 
                                    Rover.pos[1], Rover.yaw, 
                                    Rover.worldmap.shape[0], scale)
        Rover.worldmap[yWorld, xWorld, 1] += 1

        # Convert to polar coords
        distances, angles = to_polar_coords(xpix, ypix) 

        Rover.nav_dists = distances
        Rover.nav_angles = angles
    else:
        Rover.rock_detected = False
        Rover.rock_detected_first_time = True

        # Convert map image pixel values to rover-centric coords
        nav_xpix, nav_ypix = rover_coords(nav_bin_image) 
        obstacle_xpix, obstacle_ypix = rover_coords(obstacle_bin_image)  

        # Convert rover-centric pixel values to world coordinates
        scale = 10
        nav_xWorld, nav_yWorld = pix_to_world(nav_xpix, nav_ypix, Rover.pos[0],
                                    Rover.pos[1], Rover.yaw, 
                                    Rover.worldmap.shape[0], scale)
        obstacle_xWorld, obstacle_yWorld = pix_to_world(obstacle_xpix, 
                                    obstacle_ypix, Rover.pos[0], 
                                    Rover.pos[1], Rover.yaw, 
                                    Rover.worldmap.shape[0], scale)

        Rover.worldmap[obstacle_yWorld, obstacle_xWorld, 0] += 1
        Rover.worldmap[nav_yWorld, nav_xWorld, 2] += 1

        # Convert to polar coords
        distances, angles = to_polar_coords(nav_xpix, nav_ypix)

        Rover.nav_dists = distances
        Rover.nav_angles = angles
    
    
    return Rover