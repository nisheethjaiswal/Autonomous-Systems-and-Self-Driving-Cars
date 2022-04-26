import glob
import os
import sys
import random
import time
import numpy as np
import cv2

# Add to PATH (Windows only)
try:
	print("Getting Carla egg...")
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



actor_list = []
# Start the server and interact with the world
try:
	client = carla.Client('localhost', 2000)
	client.set_timeout(10.0)
	world = client.get_world()
	blueprint_library = world.get_blueprint_library()
	
	for x in range(0,10):
		print("Spawning Tesla no." + str(x))
		# Create a Tesla
		# The below will return the results in a list: [ActorBlueprint(id=vehicle.tesla.model3,tags=[vehicle, tesla, model3])]
		# So, we take the first index [0] to get the ActorBlueprint(id=vehicle.tesla.model3,tags=[vehicle, tesla, model3])
		bp = blueprint_library.filter('model3')[0]
		# Spawn the Tesla
		spawn_point = random.choice(world.get_map().get_spawn_points())
		vehicle = world.spawn_actor(bp, spawn_point)
		# Press accelerator pedal all the way down
		vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
		# Add it to the actor list for destruction later on
		actor_list.append(vehicle)
	time.sleep(10)
finally:
	# Destroy all actors. IMPORTANT for resetting the server back to the original state!
	print('destroying actors')
	for actor in actor_list:
		actor.destroy()
	print('done.')


