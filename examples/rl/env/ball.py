import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BallEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = self.masspole + self.masscart
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = self.masspole * self.length
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates
        # self.tau = 0.1
        # self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # high = np.array(
        #     [
        #         self.x_threshold * 2,
        #         np.finfo(np.float32).max,
        #         self.theta_threshold_radians * 2,
        #         np.finfo(np.float32).max,
        #     ],
        #     dtype=np.float32,
        # )

        # self.mass = 1.0

        # self.max_force = 10.0
        # self.min_force = -10.0

        self.object_radius = 2

        self.min_pixelmove = -1.5
        self.max_pixelmove = 1.5

        self.space_x_min = 0
        self.space_y_min = 0
        # self.space_x_max = 76
        # self.space_y_max = 76
        self.space_x_max = 28
        self.space_y_max = 28
        # self.space_x_max = 80
        # self.space_y_max = 80
        self.input_num_channel = 3

        # self.target_radius = 5

        # self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Box(
            # low=self.min_force,
            # high=self.max_force,
            low=self.min_pixelmove,
            high=self.max_pixelmove,
            shape=(2,),
            # shape=(5,),
            dtype=np.float32
        )

        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # shape=(self.input_num_channel, self.input_height, self.input_width),
            shape=(self.space_y_max, self.space_x_max, self.input_num_channel),
            # shape=(self.input_height, self.input_width),
            dtype=np.uint8
        )

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        self.background_image = 255 * np.ones(
            shape=[
                self.space_x_max,
                self.space_y_max,
                self.input_num_channel],
            dtype=np.uint8)

        # x, x_dot, theta, theta_dot = self.state
        # force = self.force_mag if action == 1 else -self.force_mag
        # force_x = action[0]
        # force_y = action[1]

        # force_x = action[0][0]
        # force_y = action[0][1]

        # x_dot = action[0][0]
        # y_dot = action[0][1]

        x_pixelmove = int(round(action[0][0]))
        # Clipping
        if x_pixelmove > 1:
            x_pixelmove = 1
        elif x_pixelmove < -1:
            x_pixelmove = -1

        y_pixelmove = int(round(action[0][1]))
        # Clipping
        if y_pixelmove > 1:
            y_pixelmove = 1
        elif y_pixelmove < -1:
            y_pixelmove = -1

        # if np.argmax(action) == 0: # Do not move
        #     x_pixelmove = 0
        #     y_pixelmove = 0
        # elif np.argmax(action) == 1: # Left
        #     x_pixelmove = -1
        #     y_pixelmove = 0
        # elif np.argmax(action) == 2: # Right
        #     x_pixelmove = 1
        #     y_pixelmove = 0
        # elif np.argmax(action) == 3:  # Up
        #     x_pixelmove = 0
        #     y_pixelmove = 1
        # elif np.argmax(action) == 4:  # Down
        #     x_pixelmove = 0
        #     y_pixelmove = -1


        # print(action)
        # if action == 0:
        #     x_dot = 5
        #     y_dot = 0
        # elif action == 1:
        #     x_dot = -5
        #     y_dot = 0
        # elif action == 2:
        #     x_dot = 0
        #     y_dot = 5
        # elif action == 3:
        #     x_dot = 0
        #     y_dot = -5

        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # temp = (
        #     force + self.polemass_length * theta_dot ** 2 * sintheta
        # ) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # x_acc, y_acc = force / self.mass

        # x_acc = force_x / self.mass
        # y_acc = force_y / self.mass

        # if self.kinematics_integrator == "euler":
        #     x = x + self.tau * x_dot
        #     x_dot = x_dot + self.tau * xacc
        #     theta = theta + self.tau * theta_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        # else:  # semi-implicit euler
        #     x_dot = x_dot + self.tau * xacc
        #     x = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot

        # self.x_dot = self.x_dot + self.tau * x_acc
        # self.y_dot = self.y_dot + self.tau * y_acc
        # self.x = self.x + self.tau * self.x_dot
        # self.y = self.y + self.tau * self.y_dot

        # self.x = self.x + self.tau * x_dot
        # self.y = self.y + self.tau * y_dot

        # Move
        # self.x = self.x + int(round(x_pixelmove))
        # self.y = self.y + int(round(y_pixelmove))
        self.x = self.x + x_pixelmove
        self.y = self.y + y_pixelmove
        # Agent doesn't move if the action makes the agent out of the boundary
        # penalty = 0
        if self.space_x_min <= self.x <= self.space_x_max:
            self.x = self.x
        else:
            self.x = self.x_previous
            # penalty = -10
        if self.space_y_min <= self.y <= self.space_y_max:
            self.y = self.y
        else:
            self.y = self.y_previous
            # penalty = -10
        self.x_previous = self.x
        self.y_previous = self.y

        x = int(round(self.x))
        y = int(round(self.y))
        target_x = int(round(self.target_x))
        target_y = int(round(self.target_y))

        # self.state = (x, x_dot, theta, theta_dot)
        # if x < 0 or y < 0 or target_x < 0 or target_y < 0:
        #     print("chckpt")

        # print(x, y)

        state_image_agent = cv2.circle(
            self.background_image,
            center=(x, y),
            radius=self.object_radius,
            color=(0, 0, 255), # blue
            thickness=-1)
        state_image_target = cv2.circle(
            self.background_image,
            center=(target_x, target_y),
            radius=self.object_radius,
            color=(255, 0, 0), # red
            thickness=-1)
        # plt.imshow(self.background_image)
        # plt.show()
        normalized_state_image = self.background_image / 255.0
        # self.state = normalized_state_image
        self.state = 1.0 - normalized_state_image

        # done = bool(
        #     self.x < self.space_x_min
        #     or self.x > self.space_x_max
        #     or self.y < self.space_y_min
        #     or self.y > self.space_y_max
        #     # or theta < -self.theta_threshold_radians
        #     # or theta > self.theta_threshold_radians
        # )

        done = False

        # if not done:
        #     reward = 1.0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     reward = 0.0

        distance = math.sqrt((self.x-self.target_x)**2 + (self.y-self.target_y)**2)

        # Sparse reward
        # if distance < 10:
        #     reward = 1.0
        #     done = True
        # else:
        #     reward = 0.0

        # Continuous reward
        reward = 1 - (distance / self.space_x_max)
        # done = bool(distance > 30)

        # reward = 0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.background_image = 255 * np.ones(
            shape=[
                self.space_x_max,
                self.space_y_max,
                self.input_num_channel],
            dtype=np.uint8)

        # self.x = np.random.uniform(low=1, high=self.space_x_max)
        # self.y = np.random.uniform(low=1, high=self.space_y_max)
        # self.x_dot = 0
        # self.y_dot = 0
        # self.target_x = np.random.uniform(low=1, high=self.space_x_max)
        # self.target_y = np.random.uniform(low=1, high=self.space_y_max)
        self.target_x = self.space_x_max / 2.0
        self.target_y = self.space_y_max / 2.0
        self.x = self.target_x + np.random.uniform(low=-12.0, high=12.0)
        self.y = self.target_y + np.random.uniform(low=-12.0, high=12.0)
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        x = int(round(self.x))
        y = int(round(self.y))
        target_x = int(round(self.target_x))
        target_y = int(round(self.target_y))

        state_image_agent = cv2.circle(
            self.background_image,
            center=(x, y),
            radius=self.object_radius,
            color=(0, 0, 255),  # blue
            thickness=-1)
        state_image_target = cv2.circle(
            self.background_image,
            center=(target_x, target_y),
            radius=self.object_radius,
            color=(255, 0, 0),  # red
            thickness=-1)
        # plt.imshow(self.background_image)
        # plt.show()
        normalized_state_image = self.background_image / 255.0
        # self.state = normalized_state_image
        self.state = 1.0 - normalized_state_image
        self.steps_beyond_done = None

        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        # screen_width = 600
        # screen_height = 400
        # screen_width = self.space_x_max
        # screen_height = self.space_y_max

        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0

        if self.viewer is None:
            # from gym.envs.classic_control import rendering
            #
            # self.viewer = rendering.Viewer(screen_width, screen_height)
            # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            # axleoffset = cartheight / 4.0
            # cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # self.carttrans = rendering.Transform()
            # cart.add_attr(self.carttrans)
            # self.viewer.add_geom(cart)
            # l, r, t, b = (
            #     -polewidth / 2,
            #     polewidth / 2,
            #     polelen - polewidth / 2,
            #     -polewidth / 2,
            # )
            # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # pole.set_color(0.8, 0.6, 0.4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth / 2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(0.5, 0.5, 0.8)
            # self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0, carty), (screen_width, carty))
            # self.track.set_color(0, 0, 0)
            # self.viewer.add_geom(self.track)
            #
            # self._pole_geom = pole

            # plt.close()
            # plt.ion()
            # plt.clf()
            plt.imshow(self.background_image)
            plt.show(block=False)
            plt.pause(0.0001)

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = (
        #     -polewidth / 2,
        #     polewidth / 2,
        #     polelen - polewidth / 2,
        #     -polewidth / 2,
        # )
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        # return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
