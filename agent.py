import copy
import math
import random_walk as rw
from random import random, choices, gauss
from math import sin, cos, radians, pi
from tkinter import LAST
from collections import deque

from behavior import HonestBehavior, State
from communication import CommunicationSession
from navigation import Location, NavigationTable
import numpy as np

from utils import norm, get_orientation_from_vector, rotation_matrix, rotate


class AgentAPI:
    def __init__(self, agent):
        #self._agent = agent
        self.get_mu = agent.noise_mu
        self.speed = agent.speed
        self.carries_food = agent.carries_food
        self.get_vector = agent.get_vector
        self.get_levi_turn_angle = agent.get_levi_turn_angle


    # def speed(self):
    #     return self._agent._speed
    #
    # def carries_food(self):
    #     return self._agent._carries_food
    #
    # def set_vector(self, location: Location):
    #     if location == Location.FOOD:
    #         self._agent.set_food_vector()
    #     elif location == Location.NEST:
    #         self._agent.set_nest_vector()
    #
    # def get_levi_turn_angle(self):
    #     return self._agent.get_levi_turn_angle()


def sample_noise(noise_mu, noise_musd):
    mu = gauss(noise_mu, noise_musd)
    return mu


class Agent:
    colors = {State.EXPLORING: "blue", State.SEEKING_FOOD: "orange", State.SEEKING_NEST: "green"}

    def __init__(self, robot_id, x, y, speed, radius,
                 noise_mu, noise_musd, noise_sd, environment):
        self.id = robot_id
        self.pos = np.array([x, y]).astype('float64')
        self._speed = speed
        self.radius = radius
        self.orientation = random() * 360  # 360 degree angle
        self.noise_mu = sample_noise(noise_mu, noise_musd)
        self.noise_sd = noise_sd
        self.environment = environment
        self.reward = 0

        self.levi_counter = 1
        self.trace = deque(self.pos, maxlen=100)
        self.behavior = HonestBehavior()
        self._carries_food = False
        self.api = AgentAPI(self)


    def __str__(self):
        return f"ID: {self.id}\n" \
               f"state: {self.behavior.state}\n" \
               f"expected food at: ({round(self.pos[0] + self.behavior.navigation_table.get_location_vector(Location.FOOD)[0])}, {round(self.pos[1] + self.behavior.navigation_table.get_location_vector(Location.FOOD)[1])}), \n" \
               f"   known: {self.behavior.navigation_table.is_location_known(Location.FOOD)}\n" \
               f"expected nest at: ({round(self.pos[0] + self.behavior.navigation_table.get_location_vector(Location.NEST)[0])}, {round(self.pos[1] + self.behavior.navigation_table.get_location_vector(Location.NEST)[1])}), \n" \
               f"   known: {self.behavior.navigation_table.is_location_known(Location.NEST)}\n" \
               f"info quality: \n" \
               f"   -food={round(self.behavior.navigation_table.get_target(Location.FOOD).decaying_quality, 3)}\n" \
               f"   -nest={round(self.behavior.navigation_table.get_target(Location.NEST).decaying_quality, 3)}\n" \
               f"info age:\n" \
               f"   -food={round(self.behavior.navigation_table.get_target(Location.FOOD).age, 3)}\n" \
               f"   -nest={round(self.behavior.navigation_table.get_target(Location.NEST).age, 3)}\n" \
               f"carries food: {self._carries_food}\n" \
               f"drift: {round(self.noise_mu, 5)}\n" \
               f"reward: {self.reward}$\n" \
               f"dr: {self.behavior.get_dr()}\n"

    def __repr__(self):
        return f"bot {self.id}"

    def step(self):
        self.behavior.navigation_table = self.new_nav
        sensors = self.environment.get_sensors(self)
        self.behavior.step(sensors, AgentAPI(self))
        self.move()
        self.update_trace()
        self.check_food(sensors)

    def communicate(self, neighbors):
        self.previous_nav = copy.deepcopy(self.behavior.navigation_table)
        session = CommunicationSession(self, neighbors)
        self.behavior.communicate(session)
        self.new_nav = self.behavior.navigation_table
        self.behavior.navigation_table = self.previous_nav

    def get_target(self, location):
        return self.behavior.navigation_table.get_target(location)

    def get_target_price(self, location):
        return 0

    def get_nav_location_age(self, location):
        return self.behavior.navigation_table.get_age(location)

    def get_location_vector(self, location):
        return self.behavior.navigation_table.get_location_vector(location)

    def knows_location(self, location):
        return self.behavior.navigation_table.is_location_known(location)

    def update_trace(self):
        self.trace.appendleft(self.pos[1])
        self.trace.appendleft(self.pos[0])

    def is_stationary(self):
        trace_length = len(self.trace) // 2
        tolerance = 3
        if trace_length < 4 * tolerance:  # If robot hasn't done 2*tolerance steps yet, consider it is moving
            return False
        trace_x = [self.trace[2 * i] for i in range(min(trace_length, 4 * tolerance))]
        trace_y = [self.trace[2 * i + 1] for i in range(min(trace_length, 4 * tolerance))]
        max_x_gap = max(trace_x) - min(trace_x)
        max_y_gap = max(trace_y) - min(trace_y)

        return max_x_gap < tolerance * self._speed and max_y_gap < tolerance * self._speed

    def get_vector(self, location: Location):
        if self.environment.get_sensors(self)[location]:
            return rotate(self.environment.get_location(location) - self.pos, -self.orientation)
        else:
            raise Exception(f"Robot does not sense {location}")

    def move(self):
        wanted_movement = rotate(self.behavior.get_dr(), self.orientation)
        noise_angle = gauss(self.noise_mu, self.noise_sd)
        noisy_movement = rotate(wanted_movement, noise_angle)
        self.orientation = get_orientation_from_vector(noisy_movement)
        self.pos = self.clamp_to_map(self.pos + noisy_movement)

    def clamp_to_map(self, new_position):
        if new_position[0] < self.radius:
            new_position[0] = self.radius
        if new_position[1] < self.radius:
            new_position[1] = self.radius
        if new_position[0] > self.environment.width - self.radius:
            new_position[0] = self.environment.width - self.radius
        if new_position[1] > self.environment.height - self.radius:
            new_position[1] = self.environment.height - self.radius
        return new_position

    def update_levi_counter(self):
        self.levi_counter -= 1
        if self.levi_counter <= 0:
            self.levi_counter = choices(range(1, rw.get_max_levi_steps() + 1), rw.get_levi_weights())[0]

    def get_levi_turn_angle(self):
        angle = 0
        if self.levi_counter <= 1:
            angle = choices(np.arange(0, 360), rw.get_crw_weights())[0]
        self.update_levi_counter()
        return angle

    def draw(self, canvas):
        circle = canvas.create_oval(self.pos[0] - self.radius,
                                    self.pos[1] - self.radius,
                                    self.pos[0] + self.radius,
                                    self.pos[1] + self.radius,
                                    fill=self.colors[self.behavior.state])
        self.draw_comm_radius(canvas)
        self.draw_goal_vector(canvas)
        self.draw_orientation(canvas)
        self.draw_trace(canvas)

    def draw_trace(self, canvas):
        tail = canvas.create_line(*self.trace)

    def draw_comm_radius(self, canvas):
        circle = canvas.create_oval(self.pos[0] - self.environment.robot_communication_radius,
                                    self.pos[1] - self.environment.robot_communication_radius,
                                    self.pos[0] + self.environment.robot_communication_radius,
                                    self.pos[1] + self.environment.robot_communication_radius,
                                    outline="gray")

    def draw_goal_vector(self, canvas):
        arrow = canvas.create_line(self.pos[0],
                                   self.pos[1],
                                   self.pos[0] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.FOOD),
                                       self.orientation)[0],
                                   self.pos[1] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.FOOD),
                                       self.orientation)[1],
                                   arrow=LAST)
        arrow = canvas.create_line(self.pos[0],
                                   self.pos[1],
                                   self.pos[0] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.NEST),
                                       self.orientation)[0],
                                   self.pos[1] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.NEST),
                                       self.orientation)[1],
                                   arrow=LAST)

    def draw_orientation(self, canvas):
        line = canvas.create_line(self.pos[0],
                                  self.pos[1],
                                  self.pos[0] + self.radius * cos(radians(self.orientation)),
                                  self.pos[1] + self.radius * sin(radians(self.orientation)),
                                  fill="white")

    def check_food(self, sensors):
        if self._carries_food and sensors[Location.NEST]:
            self._carries_food = False
            self.reward += 1
        if sensors[Location.FOOD]:
            self._carries_food = True

    def speed(self):
        return self._speed

    def carries_food(self):
        return self._carries_food
