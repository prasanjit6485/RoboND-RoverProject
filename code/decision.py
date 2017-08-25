import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None and \
       (Rover.pitch < 1.0 or Rover.pitch > 359.0) and \
       (Rover.roll < 1.0 or Rover.roll > 359.0):

        if Rover.rock_detected:
            if Rover.rock_detected_first_time:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.rock_detected_first_time = False
                if np.mean(Rover.nav_angles * 180/np.pi) <= 0:
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
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                # Maintain the throttle limit and steer in the direction 
                # of Rock sample when Rover vel < 0.5
                elif Rover.vel <= 0.5:
                    Rover.throttle = 0.1
                    Rover.brake = 0
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)

            # Check if Rover's near Rock sample
            # if (np.mean(Rover.nav_dists) < 8) and Rover.near_sample:
            if Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

        else:
            # Check for Rover.mode status
            if Rover.mode == 'forward':

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
                    # Set steering angle such that Rover moves close to wall
                    if Rover.vel > 0.2:
                        # Move Rover close to right side of wall
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) - 12
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

                        # print("Pos ratio: %s , Neg ratio: %s" % (posratio, negratio))

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
    
    return Rover

