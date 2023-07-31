import os
import csv
import json
import matplotlib.path as mpltPath
import numpy as np

class LandedObj:
    def __init__(self, t, volume, disapp_coeff):
        self.arrivalTime = t
        self.initial_volume = volume
        self.disapp_coeff = disapp_coeff
        self.volume = 1

    def disapperance(self, t):
        self.volume = self.initial_volume - self.disapp_coeff * (t - self.arrivalTime)
        self.volume = max(0, self.volume)

class Land:
    def __init__(self, friction, bounciness, category, url,
                 bounce_coeff, fric_coeff, revert_rate, # global warming will change these coefficients.
                 disapp_coeff,
                 position=(0, 0, 0), poly=[[]]):
        self.meta_data = {'id': url[-2:],
                          'init_friction': friction,
                          'init_bounciness': bounciness,
                          'bounce_coeff': bounce_coeff,
                          'fric_coeff': fric_coeff,
                          'revert_rate': revert_rate,
                          'disapp_coeff': disapp_coeff,
                          'category': category,
                          'position': tuple(position),
                          'center': (0, 0, 0),
                          'x_range': (),
                          'z_range': (),
                          'front': (),
                          'back': (),
                          'left': (),
                          'right': (),
                          'top': (),
                          'bottom': ()
                          }

        self.friction = friction
        self.bounciness = bounciness
        self.category = category # (0 land, 1 hole, 2 wall/dam)
        self.position = tuple(position) # (x, y, z)
        self.url = url

        self.poly = np.array(poly) / 2
        if self.category == 1:
            self.poly[:, 0] = - self.poly[:, 0]
            self.polypath = mpltPath.Path(self.poly)

        self.ball_arr = []
        self.ball_counter = 0
        self.obj_id = 0

        self.prop_his = [[0, 0, friction, bounciness]] # (t, ball_counter, friction, bounciness)

    def get_bound(self, om):
        """
        front/back is z
        left/right is x
        center is the object center
        Parameters
        ----------
        om: tdw object manager

        Returns
        -------

        """
    
        x_id = int(self.meta_data['id'][0])
        y_id = int(self.meta_data['id'][1])
        
        p1 = (4-2*x_id, -4+2*y_id)
        p2 = (6-2*x_id, -6+2*y_id)
        
        self.meta_data['center'] = ((p1[0]+p2[0])/2, 0, (p1[1]+p2[1])/2)
        self.meta_data['x_range'] = sorted((p1[0], p2[0]))
        self.meta_data['z_range'] = sorted((p2[1], p1[1]))
        print(self.meta_data['id'])
        print(self.meta_data['x_range'], self.meta_data['z_range'])
        print(self.poly)
            
        # self.meta_data['front'] = om.bounds[self.obj_id].front.tolist()
        # self.meta_data['back'] = om.bounds[self.obj_id].back.tolist()
        # self.meta_data['left'] = om.bounds[self.obj_id].left.tolist()
        # self.meta_data['right'] = om.bounds[self.obj_id].right.tolist()
        # self.meta_data['center'] = om.bounds[self.obj_id].center.tolist()
        # self.meta_data['top'] = om.bounds[self.obj_id].top.tolist()
        # self.meta_data['bottom'] = om.bounds[self.obj_id].bottom.tolist()
        # 
        # print('center',
        #       self.meta_data['center'])
        # 
        # self.meta_data['x_range'] = (round(min(self.meta_data['left'][0], self.meta_data['right'][0]), 2),
        #                              round(max(self.meta_data['left'][0], self.meta_data['right'][0]), 2))
        # self.meta_data['z_range'] = (round(min(self.meta_data['front'][2], self.meta_data['back'][2]), 2),
        #                              round(max(self.meta_data['front'][2], self.meta_data['back'][2]), 2))

    def ball_decrement(self, t): # t is frame number
        to_remove = []
        self.ball_counter = 0
        for i, ball in enumerate(self.ball_arr):
            ball.disapperance(t)
            if ball.volume == 0:
                to_remove.append(i)
            self.ball_counter += ball.volume
        for i in to_remove:
            self.ball_arr.pop(i)

    def ball_increment(self, ball):
        self.ball_arr.append(ball)

    def physical_property_change(self, t): # only related to ball accumulation
        self.ball_decrement(t)

        if self.ball_counter == 0: # when a ball disappear (evap (no effect to land), penetrate (has effect to land))
            self.friction -= self.meta_data['revert_rate'] * self.meta_data['fric_coeff'] # use a higher speed to revert
            self.bounciness += self.meta_data['revert_rate'] * self.meta_data['bounce_coeff']
            self.friction = max(self.meta_data['init_friction'], self.friction)
            self.bounciness = min(self.meta_data['init_bounciness'], self.bounciness)
        else:
            self.friction += self.ball_counter * self.meta_data['fric_coeff']
            self.bounciness -= self.ball_counter * self.meta_data['bounce_coeff']
            self.friction = min(1, self.friction)
            self.bounciness = max(0, self.bounciness)

        self.prop_his.append([t, self.ball_counter, self.friction, self.bounciness])

    def save_to_file(self, path, idx):
        folder_path = os.path.join(path, 'land'+'_'+str(idx)+'_'+str(self.obj_id))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        prop_path = os.path.join(folder_path, 'property.csv')
        header = ['time', 'ball_counter', 'friction', 'bounciness']
        with open(prop_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerows(self.prop_his)

        meta_path = os.path.join(folder_path, 'meta_data.json')
        with open(meta_path, "w") as outfile:
            json.dump(self.meta_data, outfile)


class RealObj:
    def __init__(self):
        self.position = [] # (t, x, y, z) t is global_t
        self.rotation = [] # (t, x, y, z)
        self.forward = [] # (t, x, y, z)

        self.velocity = [] # (t, x, y, z)
        self.angular_velocity = [] # (t, x, y, z)

        self.meta_data = {'obj_id': 0,
                          'sim_id': 0,
                          'start_pos': (),
                          'final_pos': (),
                          'category_final_pos': 0,
                          'start_t': 0,
                          'end_t': 0,
                          'duration': 0,
                          'apply_wind': False,
                          'final_land_id': 'xx',
                          'collision_num': 0,
                          'wind_magnitude': (0,0,0,0)} # (t, x, y, z), t is the time wind starts being applied
        self.obj_id = 0
        self.isCollide = False

    def save_to_file(self, path, sim_id, obj_idx):
        folder_path = os.path.join(path, 'ball'+'_sim_'+str(sim_id),
                                   str(obj_idx)+'_'+str(self.obj_id))

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pos_path = os.path.join(folder_path, 'position.csv')
        with open(pos_path, 'w') as f:
            write = csv.writer(f)
            write.writerows(self.position)

        rot_path = os.path.join(folder_path, 'rotation.csv')
        with open(rot_path, 'w') as f:
            write = csv.writer(f)
            write.writerows(self.rotation)

        forw_path = os.path.join(folder_path, 'forward.csv')
        with open(forw_path, 'w') as f:
            write = csv.writer(f)
            write.writerows(self.forward)

        vel_path = os.path.join(folder_path, 'velocity.csv')
        with open(vel_path, 'w') as f:
            write = csv.writer(f)
            write.writerows(self.velocity)

        ang_path = os.path.join(folder_path, 'angular_velocity.csv')
        with open(ang_path, 'w') as f:
            write = csv.writer(f)
            write.writerows(self.angular_velocity)

        meta_path = os.path.join(folder_path, 'meta_data.json')
        with open(meta_path, "w") as outfile:
            json.dump(self.meta_data, outfile)
