import carla
import gym
from gym import spaces
import numpy as np
import cv2
import math
import time
import random


SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

FIXED_DELTA_SECONDS = 0.2

SHOW_PREVIEW = True

class CarlaEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4

    #Custom Environment that follows gym interface

    def __init__(self, host='localhost', port=2000):
            # Define action and observation space
            # They must be gym.spaces objects
            #descrete actions
        self.action_space = spaces.Discrete(11*4)
        # 9 discrete for 9 different steering angles
        #4 for throttle/braking

            
        self.observation_space = spaces.Box(low=0, high=1.0,
                                            shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
                
        self.client = carla.Client(host, port)
        self.client.set_timeout(3.0)
        self.world = self.client.get_world()

            
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def cleanup(self):

        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()


    def step(self, action):
        self.step_counter +=1

            # Decode the single integer action into steer and throttle actions
        steer_index = action // 4  # Decodes to one of the 11 steering actions
        throttle_index = action % 4  # Decodes to one of the 4 throttle actions 

        # Mapping steer_index back to actual steer values
        steer_values = [-0.9, -0.40, -0.25, -0.1 , -0.05, 0.0, 0.05, 0.1, 0.25, 0.40, 0.9]
        steer = steer_values[steer_index]

        # Mapping throttle_index back to actual throttle and brake values
        if throttle_index == 0:
            throttle, brake = 0.0, 1.0
        elif throttle_index == 1:
            throttle, brake = 0.3, 0.0
        elif throttle_index == 2:
            throttle, brake = 0.7, 0.0
        elif throttle_index == 3:
            throttle, brake = 1.0, 0.0

        # Apply the decoded actions to the vehicle
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))


        #multi-discrete doesn't work for dqn
        # steer = action[0]
        # throttle = action[1]
        # #mapping all 9 steering actions
        # if steer == 0:
        #     steer = -0.9
        # elif steer == 1:
        #     steer = -0.25
        # elif steer == 2:
        #     steer = -0.1
        # elif steer == 3:
        #     steer = -0.05
        # elif steer == 4:
        #     steer = 0.0 
        # elif steer == 5:
        #     steer = 0.05
        # elif steer == 6:
        #     steer = 0.1
        # elif steer == 7:
        #     steer = 0.25
        # elif steer == 8:
        #     steer = 0.9
        # #mapping all 4 throttle/braking actions
        # if throttle == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0))
        # elif throttle == 1:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=steer, brake=0.0))
        # elif throttle == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=steer, brake=0.0))
        # else:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steer, brake=0.0))

        #prints inputs every 50 steps
        if self.step_counter % 50 == 0:
            print('Steer input from model:',steer,',throttle: ',throttle)

        #gets velocity
        v = self.vehicle.get_velocity()
        #calculates km/h
        kmh = int(3.6 *math.sqrt(v.x**2 +v.y**2 +v.z**2))

        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        #creates camera
        cam = self.front_camera

        if self.SHOW_CAM:
            cv2.imshow('Camera', cam)
            cv2.waitKey(1)

        #prevents the vehicle from chasing its tail
        lock_duration = 0
        if self.steering_lock == False:
            if steer<-0.6 or steer>0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if steer<-0.6 or steer>0.6:
                lock_duration = time.time() - self.steering_lock_start

        #rewards 
        reward = 0
        done = False
        #punish for collision
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 350
            self.cleanup()
        #lock ups
        if lock_duration > 3:
            reward = reward - 150
            done = True
            self.cleanup()
        elif lock_duration > 1:
            reward = reward - 50
        #acceleration
        #too slow
        if kmh < 5:
            reward = reward - 6
        elif kmh < 15:
            reward = reward - 3
        #too fast
        elif kmh > 45: 
            reward = reward - 10
        else: 
            reward = reward + 1
        if distance_travelled < 30:
            reward = reward - 1
        elif distance_travelled < 50:
            reward = reward + 1
        else:
            reward = reward + 2
        #checks for episode duration
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            self.cleanup()
        return cam/255.0, reward, done, {}	#curly brackets - empty dictionary required by SB3 format

    #reset everything back to normal
    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
		
        self.vehicle = None
        while self.vehicle is None:
            try:
        # connect
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"90")

		
        camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)
		# showing camera at the spawn point
        if self.SHOW_CAM:
            cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Sem Camera', self.front_camera)
            cv2.waitKey(1)
        colsensor = self.blueprint_library.find("sensor.other.collision")
        try:
            self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        except carla.CarlaException as e:
            print(f"Error: Unable to spawn colsensor actor. Reason: {e}")
            exit()
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
		
        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None # this is to count time in steering lock and start penalising for long time in steering lock
        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera/255.0


    def process_img(self, image):
        #converts image to cityscapespallete for semantic segmenatation
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

    def collision_data(self, event):
        self.collision_hist.append(event)
	
    def seed(self, seed=None):
        pass