import glob
import os
import sys
import random
import time
import numpy as np
import cv2

IM_WIDTH = 256
IM_HEIGHT = 256

# Add to PATH (Windows only)
try:
	print("Getting Carla egg...")
	# This code uses Carla 0.9.5 but most versions should work
	path = glob.glob('G:\carla\carla\PythonAPI\carla\dist\carla-0.9.5-py3.7-win-amd64.egg')[0]
	sys.path.append(path)
	print("Done.")
except:
	print("Error getting Carla egg!")
	
try:
	print("Importing Carla...")
	import carla
	print("Done.")
except:
	print("Error importing Carla!")


def process_img(image):
    i = np.array(image.raw_data)  # Convert the data to an array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # Reshape it back to an image - it has a 4th Alpha channel
    i3 = i2[:, :, :3]  # Remove the alpha channel from the image
    cv2.imshow("", i3)  # Show the camera streaming from the car
    cv2.waitKey(1)
    return i3/255.0  # Return the normalised image (0-255 to 0.0-1.0, helps computer vision)


actor_list = []
# Start the server and interact with the world
try:
	client = carla.Client('localhost', 2000)
	client.set_timeout(10.0)
	world = client.get_world()
	blueprint_library = world.get_blueprint_library()
	
	print("Spawning Tesla")
	#Create a Tesla
	# The below will return the results in a list: [ActorBlueprint(id=vehicle.tesla.model3,tags=[vehicle, tesla, model3])]
	# So, we take the first index [0] to get the ActorBlueprint(id=vehicle.tesla.model3,tags=[vehicle, tesla, model3])
	bp = blueprint_library.filter('model3')[0]
	# Spawn the Tesla
	spawn_point = random.choice(world.get_map().get_spawn_points())
	vehicle = world.spawn_actor(bp, spawn_point)
	
	# Make it drive
	vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
	actor_list.append(vehicle)
	
	# Create a camera
	blueprint = blueprint_library.find('sensor.camera.rgb')
    # Set the attributes of the camera
	blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
	blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
	blueprint.set_attribute('fov', '110')
	# We want to attach it to the front of the car. +x and +z will move it forwards and up a little bit
	spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
	# Spawn the camera and attach it to our car
	sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
	# append it for later destruction
	actor_list.append(sensor)
	#listen to the camera for a live feed
	sensor.listen(lambda data: process_img(data))
	time.sleep(20)
finally:
	print('destroying actors')
	for actor in actor_list:
		actor.destroy()
	print('done.')


