import numpy as np
import time
from datetime import datetime

def rover_home_step(Rover):
    # Update Rover's home position
    print("Update Rover's home position")
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    print("Rover started exploring at %s" % timestamp)
    Rover.home_pos = Rover.pos
    Rover.prev_stuck_pos = Rover.pos
    Rover.avoid_stuck_pos = Rover.pos
    Rover.home_state = False

    return Rover

def rover_stuck_step(Rover):
    # Check Rover is stuck at same position for more than 5 sec
    if ((time.time() - Rover.stuck_counter) > 5) and Rover.avoid_stuck:
        if (np.absolute(Rover.prev_stuck_pos[0] - Rover.pos[0]) < 0.3) and \
            (np.absolute(Rover.prev_stuck_pos[1] - Rover.pos[1]) < 0.3):
            print("Rover stuck")
            Rover.prev_stuck_yaw = Rover.yaw
            Rover.mode = 'stuck'
        Rover.prev_stuck_pos = Rover.pos
        Rover.stuck_counter = time.time()

    # Set Rover.avoid_stuck to True when Rover's moves away from home
    # position
    if (np.absolute(Rover.pos[0] - Rover.avoid_stuck_pos[0]) > 1) and \
        (np.absolute(Rover.pos[1] - Rover.avoid_stuck_pos[1]) > 1):
        Rover.avoid_stuck = True

    return Rover

# ToDo: Not working properly, need to implement fitting or RANSAC based
def rover_circular_step(Rover):
    # Check Rover is moving in circular
    if((time.time() - Rover.circular_counter) > 10) and Rover.avoid_circular:
        if np.round(Rover.prev_circular_vel,1) == np.float32(Rover.max_vel):
            print("First step done : %s" % np.round(Rover.prev_circular_vel,1))
            print("%s,%s" %(Rover.pos[0],Rover.pos[1]))
            # Set true to check whether rover is moving in circular motion
            Rover.check_circular = True
            Rover.prev_circular_pos = Rover.pos
            Rover.avoid_circular_second_step = False
        else:
            Rover.check_circular = False
        Rover.prev_circular_vel = Rover.vel
        Rover.circular_counter = time.time()

    if Rover.check_circular:
        # print("Check Rover is in circular ?")
        if (np.absolute(Rover.pos[0] - Rover.prev_circular_pos[0]) > 0.2) and \
            (np.absolute(Rover.pos[1] - Rover.prev_circular_pos[1]) > 0.2):
            Rover.avoid_circular_second_step = True
        if (np.absolute(Rover.pos[0] - Rover.prev_circular_pos[0]) < 0.2) and \
            (np.absolute(Rover.pos[1] - Rover.prev_circular_pos[1]) < 0.2) and\
            Rover.avoid_circular_second_step:
            print("Rover in Circular motion")
            Rover.mode = 'circular'
            Rover.check_circular = False
            Rover.avoid_circular_second_step = False
            # Rover.circular_counter = time.time()
    
    # Set Rover.avoid_stuck to True when Rover's moves away from home
    # position
    if (np.absolute(Rover.pos[0] - Rover.home_pos[0]) > 1) and \
        (np.absolute(Rover.pos[1] - Rover.home_pos[1]) > 1):
        Rover.avoid_circular = True

    return Rover

def rock_sample_step(Rover):
    # Check if all samples are collected 
    if Rover.samples_collected == 6:
        if (np.absolute(Rover.pos[0] - Rover.home_pos[0]) < 5) and \
            (np.absolute(Rover.pos[1] - Rover.home_pos[1]) < 5):
            Rover.mode = 'return_home'
    
    return Rover

def limit_rover_max_vel_step(Rover, max_vel = 1200.0):
    # Limit Rover's max velocity to 1 after 20 mins
    if (np.round(Rover.total_time, 1) > max_vel) and Rover.max_state:
        print("Limit Rover's maximum velocity to 1m/s")
        Rover.max_vel = 1
        Rover.max_state = False

    return Rover

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Apply 4-wheel turning for 45 degree when Rover is stuck 
    if Rover.mode == 'stuck':
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = 0
        Rover.steer = 15 # left direction, since wall crawling is right side   
        # If Rover rotated more than 45 degree release from stuck mode
        if np.absolute(Rover.yaw - Rover.prev_stuck_yaw) > 45:
            Rover.mode = 'forward'
        return Rover

    # Move Rover in the left direction, since Rover will move in 
    # clock-wise direction due to right side wall crawling
    if Rover.mode == 'circular':
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = 0
        Rover.steer = 15 # left direction, since wall crawling is right side
        if (np.absolute(Rover.pos[0] - Rover.prev_circular_pos[0]) > 1) and \
            (np.absolute(Rover.pos[1] - Rover.prev_circular_pos[1]) > 1):
            Rover.mode = 'forward'
        return Rover

    # If returned home, do nothing
    if Rover.mode == 'return_home':
        if Rover.end_state:
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            print("Returned home at %s" % timestamp)
            print("Rover took %s sec to complete the task" % np.round(Rover.total_time, 1))
            Rover.end_state = False
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.avoid_stuck = False
        return Rover

    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        if Rover.rock_detected:
            # print("Rock detected")
            if Rover.rock_detected_first_time:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = np.clip(np.mean(Rover.rock_nav_angles * 180/np.pi), -15, 15)
                Rover.rock_detected_first_time = False
                if np.mean(Rover.rock_nav_angles * 180/np.pi) <= 0:
                    Rover.four_wheel_turn = 15 # Rock detected right side
                else:
                    Rover.four_wheel_turn = -15 # Rock detected left side
            else:
                # Keep Rover moving in the direction of Rock samples
                # Apply full brake and steer in the direction 
                # of Rock sample when Rover vel > 0.5
                if Rover.vel > 0.5:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = np.clip(np.mean(Rover.rock_nav_angles * 180/np.pi), -15, 15)
                # Maintain the throttle limit and steer in the direction 
                # of Rock sample when Rover vel < 0.5
                elif Rover.vel <= 0.5:
                    # If Rover stuck near wall while collecting Rock sample, 
                    # full throttle
                    if Rover.vel < 0.2:
                        Rover.throttle = 1.0
                    else:
                        Rover.throttle = 0.1
                    Rover.brake = 0
                    Rover.steer = np.clip(np.mean(Rover.rock_nav_angles * 180/np.pi), -15, 15)

            # Check if Rover's near Rock sample
            if Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

        else:
            # Check for Rover.mode status
            if Rover.mode == 'forward':

                # Detect obstacles from truncate navigable distance and
                # when Rover is stable within 1 degree of roll and pitch and
                # when Rover's velocity > 0.2
                if np.mean(Rover.trun_nav_dists) < 10 and \
                    (Rover.pitch < 1.0 or Rover.pitch > 359.0) and \
                    (Rover.roll < 1.0 or Rover.roll > 359.0) and \
                    Rover.vel > 0.2:
                    print("Obstacle, Apply brakes")
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

                else:

                    # Check the extent of navigable terrain
                    # if there's enough navigable terrain pixels then go forward
                    if len(Rover.nav_angles) >= Rover.stop_forward:  
                        # If mode is forward, navigable terrain looks good 
                        # and velocity is below max, then throttle 
                        if Rover.vel < Rover.max_vel:
                            # Set throttle value to throttle setting
                            Rover.throttle = Rover.throttle_set
                        else: # Else coast
                            Rover.throttle = 0
                        Rover.brake = 0
                        # Set steering to average angle clipped to the range +/- 15
                        # Set steering angle such that Rover moves close to 
                        # wall but avoid narrow navigable terrain
                        # and when Rover velocity > 0.2
                        if Rover.vel > 0.2 and len(Rover.nav_angles) > 1500:
                            # print("Nav pixels:%s" % len(Rover.nav_angles))
                            # Move Rover close to right side of wall
                            Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) - 14
                        else:
                            Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        
                        # Check if Rover stuck near wall induce 4-wheel turning
                        if Rover.vel == 0.0:

                            # Compute mean of positive angles and negative angles
                            # and induce 4-wheel turning in the direction of 
                            # navigable terrain (more positive angles, go left)
                            angles = Rover.nav_angles
                            negratio = float(len(angles[angles < 0]))/len(angles)
                            posratio = float(len(angles[angles > 0]))/len(angles)

                            # Induce 4-wheel turning in the left direction
                            if posratio >= 0.6:
                                # print("stuck, Rotate left")
                                Rover.throttle = 0
                                # Release the brake to allow turning
                                Rover.brake = 0
                                Rover.steer = 15
                            # Induce 4-wheel turning in the right direction
                            elif negratio >= 0.6:
                                # print("Stuck, Rotate right")
                                Rover.throttle = 0
                                # Release the brake to allow turning
                                Rover.brake = 0
                                Rover.steer = -15
                            # If Rover stuck near wall and not inducing 
                            # 4-wheel turn, full throttle
                            else:
                                Rover.throttle = 1.0

                    # If there's a lack of navigable terrain pixels then go to
                    # 'stop' mode
                    elif len(Rover.nav_angles) < Rover.stop_forward:
                            # Set mode to "stop" and hit the brakes!
                            Rover.throttle = 0
                            # Set brake to stored brake value
                            Rover.brake = Rover.brake_set
                            Rover.steer = 0
                            Rover.mode = 'stop'

            # If we're already in "stop" mode then make different decisions
            elif Rover.mode == 'stop':
                
                # If we're in stop mode but still moving keep braking
                if Rover.vel > 0.2:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                # If we're not moving (vel < 0.2) then do something else
                elif Rover.vel <= 0.2:
                    # Now we're stopped and we have vision data to see if there's a path forward
                    if len(Rover.nav_angles) < Rover.go_forward:
                        Rover.throttle = 0
                        # Release the brake to allow turning
                        Rover.brake = 0
                        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                        Rover.steer = Rover.four_wheel_turn # Could be more clever here about which way to turn
                        Rover.four_wheel_turn = 15
                    # If we're stopped but see sufficient navigable terrain in front then go!
                    if len(Rover.nav_angles) >= Rover.go_forward:
                        # Set throttle back to stored value
                        Rover.throttle = Rover.throttle_set
                        # Release the brake
                        Rover.brake = 0
                        # Set steer to mean angle
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        Rover.mode = 'forward'

    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        # Avoid stuck mode after pickup
        Rover.avoid_stuck_pos = Rover.pos
        Rover.avoid_stuck = False
    
    return Rover

