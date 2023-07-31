import argparse
import os
import shutil

import numpy as np
import yaml
from tdw.add_ons.collision_manager import CollisionManager
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils

from sim_objects import LandedObj, Land, RealObj

# commands necessary for changing static properties in object manager
STATIC_PROP_READ = [{"$type": "send_static_rigidbodies",
                     "frequency": "once"},
                    {"$type": "send_segmentation_colors",
                     "frequency": "once"},
                    {"$type": "send_bounds",
                     "frequency": "once"},
                    {"$type": "send_categories",
                     "frequency": "once"}]

# Get the random number generator
rng = np.random.RandomState()

class Processor:
    '''
    In TDW, (x, z) are horizontal dimensions and y is height.
    '''

    def __init__(self, config_path, port=1071):
        with open(r'{}'.format(config_path)) as f:
            args = yaml.full_load(f)

        self.c = Controller(port=port) #(launch_build=False, port=port)  # set the controller
        self.lands = []   # list to save land obj
        self.land_xzpositions = []   # list to save [idx, x, z] of each land
        self.global_t = 0  # global time/frame variable
        self.land_x_bound = ()  #
        self.land_z_bound = ()
        self.hole_dam_pos = []
        self.accele_th = 0.015  # the threshold of minimum acceleration to determine if a ball becomes static
        self.vec_th = 0.50  # the threshold of minimum velocity to determine if a ball becomes static

        self.output_path = args['output_path']  # output path
        # set configs from file.
        self.max_frame = args['max_frame']
        self.num_sim = args['num_sim']  # number of simulation
        self.rand_sleep_interval = args['rand_sleep_interval']  # the range of sleep interval between two consecutive dropping balls.
        self.wind = args['apply_wind']  # bool, whether to apply wind
        self.wind_mag = args['wind_mag']    # wind magnitude
        self.wind_torq = args['wind_torq']  # wind torque
        self.y_range = args['height_range']

        self.create_env()
        print("Created Environemnt")
        # Uncomment these lines to enable cameras for simulation visualization
        # but will make the speed much slower:
        # self.set_cameras()
        # print("set cameras")
        # self.set_cam_focus()
        # print('set focus')
        self.set_obj_manager()
        print("set object manager")
        self.set_collision_manager()
        print("set collision manager")

        self.init_lands(args['params'])
        print("initialized lands")
        self.create_lands()
        print("created lands")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        shutil.copyfile(config_path, os.path.join(self.output_path, config_path.split('/')[1]))

    def set_cameras(self):
        # set up camera capture
        self.cameras = ['a', 'b', 'c']
        camera_a = ThirdPersonCamera(avatar_id="a", #ground
                                     position={"x": 3, "y": 2, "z": 3},
                                     look_at={"x": -2, "y": 0, "z": -2})
        camera_b = ThirdPersonCamera(avatar_id="b", #tree
                                     position={"x": 2, "y": 7, "z": 2},
                                     look_at={"x": 0, "y": 7, "z": 0})

        camera_c = ThirdPersonCamera(avatar_id="c", #overall
                                     position={"x": 0, "y": 15, "z": 0},
                                     look_at={"x": 0, "y": -1, "z": 0})

        for camera in self.cameras:
            self.c.add_ons.extend([camera_a, camera_b, camera_c])

        # set path for saving rendered images.
        # path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("image_only_video")
        # path = Path('controller_output')
        # path = path.joinpath('image_only_video')
        # print(f"Images at: {path.resolve()}")
        # self.capture = ImageCapture(path=path, avatar_ids=["a", "b", "c"], png=True)
        # self.c.add_ons.append(self.capture)

    def create_env(self):
        """
        define exterior room & initialize environment
        """
        self.c.communicate([TDWUtils.create_empty_room(30, 30),{"$type": "set_render_quality", "render_quality": 1},]) #5 is best.
                            #{"$type": "set_screen_size", "width": 1200, "height": 1200}, # other parameters set for better rendering.
                            # {"$type": "add_hdri_skybox",
                            #  "name": "bergen_4k",
                            #  "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/windows/2019.1/bergen_4k",
                            #  "exposure": 0.6,
                            #  "initial_skybox_rotation": 210,
                            #  "sun_elevation": 500,
                            #  "sun_initial_angle": 50,
                            #  "sun_intensity": 1}])
    def set_cam_focus(self):
        """
        set camera focus
        """
        self.c.communicate([{"$type": "focus_on_object", "object_id": self.lands[17].obj_id,
                             "use_centroid": True, "sensor_name": "SensorContainer", "avatar_id": "c"},
                            {"$type": "set_aperture", "aperture": 2.0},
                            {"$type": "set_ambient_intensity", "intensity": 10000.0}])

    def set_obj_manager(self):
        """
        set object manager
        """
        self.object_manager = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
        self.c.add_ons.append(self.object_manager)

    def set_collision_manager(self):
        """
        set collision manager
        """
        self.collision_manager = CollisionManager(enter=True, stay=True, exit=True, objects=True, environment=True)
        self.c.add_ons.append(self.collision_manager)

    def init_lands(self, params=[]):
        for i in range(len(params)):
            land = Land(**params[i])
            self.lands.append(land)

    def create_lands(self):
        self.land_ids = []

        for i, land in enumerate(self.lands):
            land_id = self.c.get_unique_id()
            land.obj_id = land_id
            self.land_ids.append(land_id)

            self.c.communicate([
                {"$type": "add_object",
                 "name": "land{}".format(str(i)),
                 "url": land.url,
                 "scale_factor": 0.5,
                 "position": {"x": 0, "y": 0, "z": 0},  # set land block position in sim env.
                 "id": land_id},
                {"$type": "set_kinematic_state",
                 "id": land_id,
                 "is_kinematic": True, # if True, it will not respond to physics. When True, collision can still happen,
                                        # and changing physical properties on the fly can work as well.
                 "use_gravity": True}, # if True, it will respond to gravity.
                {"$type": "set_mass",
                 "mass": 100000, # set mass high or change is_kinematic to be True to avoid the movement of lands.
                 "id": land_id},
                {"$type": "set_physic_material",
                 "dynamic_friction": land.friction,  # set physical properties.
                 "static_friction": land.friction,
                 "bounciness": land.bounciness,
                 "id": land_id}
            ])

        # get the bounds of each land block, and the total size of the entire combined land.
        min_x, max_x = 10000, -10000
        min_z, max_z = 10000, -10000
        for i, land in enumerate(self.lands):
            land.get_bound(self.object_manager)
            min_x = min(min_x, land.meta_data['x_range'][0])
            max_x = max(max_x, land.meta_data['x_range'][1])
            min_z = min(min_z, land.meta_data['z_range'][0])
            max_z = max(max_z, land.meta_data['z_range'][1])

            # find the bounds of land blocks of class 2 (wall)
            if land.category == 2:
                # print(land.url)
                # print(land.meta_data['x_range'], land.meta_data['z_range'])
                self.hole_dam_pos.append(list(land.meta_data['x_range']) + list(land.meta_data['z_range']))

            self.land_xzpositions.append([i, land.meta_data['center'][0], land.meta_data['center'][2]])

        self.land_x_bound = (min_x, max_x)
        self.land_z_bound = (min_z, max_z)
        print("land x bound: ", self.land_x_bound)
        print("land z bound: ", self.land_z_bound)
        print('hole/dam pos: ', self.hole_dam_pos)
        print('every land x z position: ', self.land_xzpositions)

    def create_frozen_apples(self, sim_id, num_obj=1):
        """
        creates balls suspended in place
        Parameters
        ----------
        sim_id
        num_obj

        Returns
        -------

        """

        self.real_ball_objs = []
        self.obj_ids = []
        commands = []
        for _ in range(num_obj):
            obj_id = self.c.get_unique_id()

            lo_height = int(self.y_range[0])
            hi_height = int(self.y_range[1])
            # ball position are in 0.1 resolution.
            position = [round(rng.randint(self.land_x_bound[0]*10, self.land_x_bound[1]*10) / 10, 1),
                        round(rng.randint(lo_height*10, hi_height*10) / 10, 1),
                        round(rng.randint(self.land_z_bound[0]*10, self.land_z_bound[1]*10) / 10, 1)]

            # new obj and set obj attributes.
            obj = RealObj()
            obj.position.append([self.global_t, position[0], position[1], position[2]])
            obj.rotation.append([self.global_t, 0, 0, 0])
            obj.forward.append([self.global_t, 0, 0, 0])
            obj.angular_velocity.append([self.global_t, 0, 0, 0])
            obj.velocity.append([self.global_t, 0, 0, 0])
            obj.obj_id = obj_id
            obj.meta_data['start_t'] = self.global_t
            obj.meta_data['obj_id'] = obj_id
            obj.meta_data['sim_id'] = sim_id
            obj.meta_data['start_pos'] = tuple(position)

            commands.extend(self.c.get_add_physics_object(model_name="prim_sphere",
                                                          object_id=obj_id,
                                                          library='models_special.json',
                                                          scale_factor={"x": 0.2, "y": 0.2, "z": 0.2},
                                                          position={"x": position[0],
                                                                    "y": position[1],
                                                                    "z": position[2]},
                                                          bounciness=0.8,
                                                          mass=0.1,
                                                          gravity=True))
            self.real_ball_objs.append(obj)
            self.obj_ids.append(obj_id)

        self.c.communicate(commands=commands)
        self.set_collision_manager()
        self.global_t += 1

    def apply_wind(self, obj):
        # windx = rng.normal(self.wind_mag[0], self.wind_mag[0]/3)
        # windy = rng.normal(self.wind_mag[1], self.wind_mag[1]/3)
        # windz = rng.normal(self.wind_mag[2], self.wind_mag[2]/3)
        windx = self.wind_mag[0]
        windy = self.wind_mag[1]
        windz = self.wind_mag[2]

        torquex = self.wind_torq[0]
        torquey = self.wind_torq[1]
        torquez = self.wind_torq[2]
        command = [{"$type": "apply_force_to_object", "id": obj.obj_id, "force": {"x": windx, "y": windy, "z": windz}},
                   {"$type": "apply_torque_to_object", "id": obj.obj_id, "force":
                       {"x": torquex, "y": torquey, "z": torquez}}]
        return command, (windx, windy, windz, torquex, torquey, torquez)

    def _update_land(self, obj, t, volume=1):
        last_t, last_x, last_z, = obj.position[-1][0], obj.position[-1][1], obj.position[-1][3] # (t, x, y, z)

        if last_x > self.land_x_bound[1] or last_x < self.land_x_bound[0] or last_z > \
                self.land_z_bound[1] or last_z < self.land_z_bound[0]:
            category = -1
            final_land_id = 'xx'
        else:
            dist_sort = sorted(self.land_xzpositions, key=lambda x: (x[1]-last_x)**2+(x[2]-last_z)**2) # (id, x, z)

            closest_land_idx = dist_sort[0][0]
            landed_ball = LandedObj(t, volume, self.lands[closest_land_idx].meta_data['disapp_coeff'])
            self.lands[closest_land_idx].ball_increment(landed_ball)
            category = self.lands[closest_land_idx].category
            final_land_id = self.lands[closest_land_idx].meta_data['id']

        obj.meta_data['final_pos'] = tuple(obj.position[-1][1:])
        obj.meta_data['end_t'] = obj.position[-1][0]
        obj.meta_data['duration'] = t - obj.meta_data['start_t']
        obj.meta_data['category_final_pos'] = category
        obj.meta_data['final_land_id'] = final_land_id

    def run_single_sim(self, sim_id, applyWind=False):
        done = False
        updated_obj = set()

        if applyWind:
            t_to_apply = [rng.randint(0, 60) for _ in range(len(self.real_ball_objs))]

        single_t = 0
        while not done and single_t < self.max_frame: # make sure all balls will contact the ground.
            # read every obj's states
            # ('global time.... ', self.global_t)
            commands = []

            tmp_done = True

            for idx, obj in enumerate(self.real_ball_objs):
                pos = self.object_manager.transforms[obj.obj_id].position
                rot = self.object_manager.transforms[obj.obj_id].rotation
                forw = self.object_manager.transforms[obj.obj_id].forward
                vec = self.object_manager.rigidbodies[obj.obj_id].velocity
                ang = self.object_manager.rigidbodies[obj.obj_id].angular_velocity
                sleeping = self.object_manager.rigidbodies[obj.obj_id].sleeping

                # early stop condition check
                early_stop = False
                # print(pos, self.collision_manager.obj_collisions, self.collision_manager.env_collisions)
                for col_obj_id in self.collision_manager.obj_collisions: # obj collision
                    # print(col_obj_id, obj.obj_id, self.collision_manager.obj_collisions[col_obj_id].state)
                    if obj.obj_id in [col_obj_id.int1, col_obj_id.int2] and \
                            self.collision_manager.obj_collisions[col_obj_id].state=='enter':
                        obj.isCollide = True
                        obj.meta_data['collision_num'] += 1
                        #   print('OBJ collision!')

                for col_obj_id in self.collision_manager.env_collisions:  # env collision
                    if col_obj_id == obj.obj_id and \
                            self.collision_manager.env_collisions[obj.obj_id].state=='enter':
                        obj.isCollide = True
                        obj.meta_data['collision_num'] += 1
                        print('ENV collision!')

                # check if the ball is out of the region
                if obj.isCollide and (pos[0] > self.land_x_bound[1] or pos[0] < self.land_x_bound[0] or pos[
                    2] > self.land_z_bound[1] or pos[2] < self.land_z_bound[0]):
                    early_stop = True
                    print("outside region")

                # check if the pos is inside a hole or a dam.
                if obj.isCollide and self.hole_dam_pos:
                    for h_xmin, h_xmax, h_zmin, h_zmax in self.hole_dam_pos: # (xmin, xmax, zmin, zmax)
                        if h_xmin<=pos[0]<= h_xmax and h_zmin<=pos[2]<=h_zmax:
                            early_stop = True
                            print('inside dam')
                if obj.isCollide:
                    for land in self.lands:
                        if land.category == 1:
                            inside = land.polypath.contains_points([[pos[0], pos[2]]])
                            if inside[0] == 1:
                                early_stop = True
                                print('inside hole')

                # check a ball's acceleration
                if obj.isCollide and len(obj.velocity) >= 10:
                    accele = []
                    vec_arr = []
                    for vec_i in range(-1, -6, -1):
                        curr_x, curr_y, curr_z = obj.velocity[vec_i][1:]
                        last_x, last_y, last_z = obj.velocity[vec_i-1][1:]

                        curr_v = ((curr_x)**2+(curr_y)**2+(curr_z)**2) ** 0.5
                        last_v = ((last_x)**2+(last_y)**2+(last_z)**2) ** 0.5

                        accele.append(abs(curr_v - last_v))
                        vec_arr.append(curr_v)
                    ave_accele = sum(accele) / len(accele)
                    ave_vec = sum(vec_arr) / len(vec_arr)
                    # print('ave, ', ave_vec, ave_accele)
                    if ave_accele <= self.accele_th and ave_vec <= self.vec_th:
                        early_stop = True
                        print('acceleration or velocity below the sleep threshold')

                if obj.obj_id not in updated_obj:
                    # pos = np.round(pos, 2)
                    obj.position.append([self.global_t] + list(pos))
                    obj.rotation.append([self.global_t] + list(rot))
                    obj.forward.append([self.global_t] + list(forw))
                    obj.velocity.append([self.global_t] + list(vec))
                    obj.angular_velocity.append([self.global_t] + list(ang))

                tmp_done = tmp_done & (sleeping | early_stop)

                if sleeping or early_stop: # Update land
                    self._update_land(obj, self.global_t)
                    updated_obj.add(obj.obj_id)
                    print("stopped: ", sleeping, early_stop)

                if applyWind and not (sleeping or early_stop):
                    if single_t == t_to_apply[idx]:
                        wind_cmd, wind_mag = self.apply_wind(obj)
                        commands.extend(wind_cmd)
                        obj.meta_data['wind_magnitude'] = tuple([self.global_t]) + wind_mag
                        obj.meta_data['apply_wind'] = True
                        # print('applying wind: ', wind_mag)

            # pil_image = self.capture.get_pil_images()['c']['_img'].convert('RGB')
            # # open_cv_image = np.array(pil_image)
            # pil_image.save("ball_drop_paper.png", dpi=(300, 300))
            # print('images:', pil_image.size)

            # every iteration, update the land physical properties
            for land in self.lands:
                land.physical_property_change(self.global_t)
                commands.append({"$type": "set_physic_material",
                                 "dynamic_friction": land.friction,
                                 "static_friction": land.friction,
                                 "bounciness": land.bounciness,
                                 "id": land.obj_id})

            commands.extend(STATIC_PROP_READ) # necessary for changing static properties in object manager
            resp = self.c.communicate(commands) # advance
            self.object_manager._cached_static_data = False # goto the object_manager source code to see why.
            self.object_manager.on_send(resp)
            self.global_t += 1
            single_t += 1
            done = tmp_done

        # when outside the loop, check the non-updated ball position
        for idx, obj in enumerate(self.real_ball_objs):
            if obj.obj_id not in updated_obj:
                self._update_land(obj, self.global_t)
            # Destroy the object.

            obj.save_to_file(self.output_path, sim_id, idx)
            self.c.communicate({"$type": "destroy_object",
                                "id": obj.obj_id})

        # Mark the object manager as requiring re-initialization.
        self.object_manager.initialized = False
        self.collision_manager.initialized = False

    def process(self):
        """
        Full simulation processor.
        Returns
        -------

        """
        for sim_id in range(self.num_sim):
            print("sim {}.......................".format(sim_id))
            self.create_frozen_apples(sim_id)
            self.run_single_sim(sim_id, self.wind)
            print('single run finish, sleeping')
            rng_interval = rng.randint(10, self.rand_sleep_interval)
            commands = []
            for t in range(self.global_t, self.global_t+rng_interval):
                # update the land properties
                for land in self.lands:
                    land.physical_property_change(t)
                    commands.append({"$type": "set_physic_material",
                                     "dynamic_friction": land.friction,
                                     "static_friction": land.friction,
                                     "bounciness": land.bounciness,
                                     "id": land.obj_id})
                commands.extend(STATIC_PROP_READ) # necessary for changing static properties in object manager

                resp = self.c.communicate(commands) # advance
                self.object_manager._cached_static_data = False # goto the object_manager source code to see why.
                self.object_manager.on_send(resp)

            self.global_t += rng_interval

        for idx, land in enumerate(self.lands):
            land.save_to_file(self.output_path, idx)

        self.c.communicate({"$type": "terminate"})
        print("sim end")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6788, required=False)
    parser.add_argument('--config_path', type=str, default='config/land_params_V3.1.yaml', required=False)
    arg = parser.parse_args()
    print(arg.port)
    proc = Processor(arg.config_path, port=arg.port)
    proc.process()
