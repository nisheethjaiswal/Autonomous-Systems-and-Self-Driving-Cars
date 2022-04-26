import glob
import os
# Suppress warnings to prevent interruption of the progress bar. Only do this after the code is tested!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from threading import Thread
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, AveragePooling2D, Conv2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

csv_file = open("G:\\Carla\\experiments\\output.csv", 'w')

# This code uses Carla 0.9.5 but most versions should work
try:
	print("Getting Carla egg...")
	# Windows only!
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

SHOW_PREVIEW = False # Whether to show camera stream
IM_WIDTH = 256 # Camera width
IM_HEIGHT = 256 # Camera height
SECONDS_PER_EPISODE = 30 # How long before episode is ended automatically
REPLAY_MEMORY_SIZE = 5_000 
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5 # Update weights per episode(s)
MODEL_NAME = "CNN" # For saving the model file
MEMORY_FRACTION = 0.4
AVG_REWARD_TO_SAVE = -10000 # This is dynamic and will update per episode
EPISODES = 10000 # How long to rubn for
DISCOUNT = 0.95 # Discount factor
epsilon = 1 # Starting epsilon (for random actions)
EPSILON_DECAY = 0.99 # Decay to epsilon to reduce random actions
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 5 # Aggregate stats, possibly save the model

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        # The below will return the results in a list: [ActorBlueprint(id=vehicle.tesla.model3,tags=[vehicle, tesla, model3])]
        # So, we take the first index [0] to get the ActorBlueprint(id=vehicle.tesla.model3,tags=[vehicle, tesla, model3])
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.ticks = 0
        self.episode_number = -1
        self.video_writer = None
    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.episode_number += 1
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.episode_number % 20 == 0:
            self.video_writer = cv2.VideoWriter("output_"+str(self.episode_number).zfill(8)+".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (IM_HEIGHT, IM_WIDTH))
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

        self.ticks += 1
        if self.video_writer is not None:
            self.video_writer.write(i3)

        # if self.ticks % 100 == 0:
            # cv2.imwrite('output.png', i3)

        # if self.SHOW_CAM:
            # cv2.imshow("camera", i3)
            # cv2.waitKey(1)
            # print('\nd')
        self.front_camera = i3

    def step(self, action):
        #go straight
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(brake=0, throttle=1.0, steer=0))
        #hard left
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(brake=0, throttle=1.0, steer=-1))
        #hard right
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(brake=0, throttle=1.0, steer=1))
        #half left
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(brake=0, throttle=1.0, steer=-0.5))
        #half right
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(brake=0, throttle=1.0, steer=0.5))
        #brakes on, straight on
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(brake=1, steer=0))
        #brakes on, half left
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(brake=1, steer=-0.5))
        #brakes on, half right
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(brake=1, steer=0.5))
        
        # Calculate speed to try and prevent the car from driving around in a circle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # Reward strategy
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif kmh < 10:
            done = False
            reward = -2
        elif kmh < 20:
            done = False
            reward = -1
        elif kmh < 30:
            done = False
            reward = 1
        elif kmh < 40:
            done = False
            reward = 2
        else:
            done = False
            reward = 3

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        with sess.as_default():
            tf.initialize_all_variables().run()

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH,3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        
        model.add(Flatten())
        
        model.add(Dense(256))
        model.add(Activation('relu'))
        
        model.add(Dense(128))
        model.add(Activation('relu'))
        
        model.add(Dense(8))
        model.add(Activation('linear'))
        
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        
        with self.graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        
        with self.graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        with self.graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 8)).astype(np.float32)
        
        with self.graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # self.model.fit(X,y, verbose=True, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 60 # Goal runs/sec for model in simulation
    ep_rewards = [-200]

    # For repeatable experiments
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    backend.set_session(sess)
    graph = tf.get_default_graph()
    
    # Check if models folder exists
    if not os.path.isdir('models'):
        os.makedirs('models') # Create a directory for saving

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    print('Waiting for agent to be initialised')
    while agent.training_initialized is False:
        time.sleep(0.01)

    # Initialize predictions
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Now let's run the training!
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            env.collision_hist = []
            
            # Reset rewards and steps back to the start
            episode_reward = 0
            step = 1

            # Reset environment
            current_state = env.reset()

            # Set done back to False and record start time (for termination after set time)
            done = False
            episode_start = time.time()

            # Train until either collision or time up
            while True:
                # Check for random occurence
                if np.random.random() > epsilon:
                    # No random occurence, get model to predict
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Generate a random action
                    action = np.random.randint(0, 8) # Update this if adding more actions!
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1./FPS)
                    
                new_state, reward, done, _ = env.step(action)
                
                # Check total rewards
                episode_reward += reward

                # Update memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                
                # Prep for next step
                current_state = new_state
                step += 1
                
                # Did we collide or run out of time? If so, we are done and must stop
                if done:
                    break
            
            # Episode now finished
            for actor in env.actor_list:
                actor.destroy()
            print("Reward: " + str(episode_reward))

            csv_file.write(','.join([str(x) for x in [reward,epsilon]])+"\n")
            csv_file.flush()

            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                print("Average Reward: " + str(average_reward))
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                
                # If the average reward is higher than previous, save model
                if average_reward >= AVG_REWARD_TO_SAVE:
                        AVG_REWARD_TO_SAVE = average_reward
                        with tf.Session() as sess:
                            sess.run(tf.global_variables_initializer())
                            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon value for lower chance of random actions 
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    # Save final model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
