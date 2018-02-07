from pyglet.window import Platform
# global variables
screen = Platform().get_default_display().get_default_screen()

# parameters
PIXELS_PER_METER = 50
LINE_WIDTH = 0.15
ROAD_WIDTH = 7.5
ORIGIN = (0, 3.75)
DESTINATION = (10, 3.75)
INITIAL_V = (0, 0)
WIDTH =  screen.width
HEIGHT =  ROAD_WIDTH * PIXELS_PER_METER # screen.height
SPEAD_LIMIT = 20 # 72 km/h
# when the car is in the circle,
# we assume it achieves the goal
DESTINATION_RADIUS = 0.3

# reward and loss coefficients
out_of_range_loss = 100
overspeed_loss = 80

distance_coeff = 10
direction_coeff = 10
acceleration_coeff = 10
jerk_coeff = 10
time_loss_coeff = 0.1

