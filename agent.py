import copy
from random import random, choices, gauss
from math import sin, cos, radians, pi
from tkinter import LAST
from enum import Enum
from collections import deque
from navigation import Location, NavigationTable
import numpy as np

from strategy import BetterAgeStrategy, WeightedAverageAgeStrategy, QualityStrategy, DecayingQualityStrategy
from utils import norm, get_orientation_from_vector, rotation_matrix


class State(Enum):
    EXPLORING = 1
    SEEKING_FOOD = 2
    SEEKING_NEST = 3


class Agent:
    thetas = np.arange(0, 360)
    max_levi_steps = 1000
    crw_weights = []
    levi_weights = []
    colors = {State.EXPLORING: "blue", State.SEEKING_FOOD: "orange", State.SEEKING_NEST: "green"}

    def __init__(self, robot_id, x, y, speed, radius, rdwalk_factor, levi_factor,
                 noise_mu, noise_musd, noise_sd, environment):
        self.id = robot_id
        self.pos = np.array([x, y]).astype('float64')
        self.speed = speed
        self.radius = radius
        self.rdwalk_factor = rdwalk_factor
        self.levi_factor = levi_factor
        self.orientation = random() * 360  # 360 degree angle
        self.displacement = np.array([0, 0]).astype('float64')
        self.noise_mu, self.noise_sd = self.sample_noise(noise_mu, noise_musd, noise_sd)
        self.environment = environment
        self.neighbors = []
        self.state = State.SEEKING_FOOD
        self.reward = 0
        self.crw_weights = self.crw_pdf(self.thetas)
        self.levi_weights = self.levi_pdf(self.max_levi_steps)
        self.levi_counter = 1
        self.trace = deque(self.pos, maxlen=100)
        self.navigation_table = NavigationTable(quality=1-abs(self.noise_mu))
        self.new_information = NavigationTable(quality=1-abs(self.noise_mu))

        # self.navigation_table.set_location_vector(Location.FOOD, self.environment.get_food_location() - self.pos)
        # self.navigation_table.set_location_vector(Location.NEST, self.environment.get_nest_location() - self.pos)
        # self.navigation_table.set_location_known(Location.FOOD, True)
        # self.navigation_table.set_location_known(Location.NEST, True)

        self.carries_food = False

    def __str__(self):
        return f"Bip boop, I am bot {self.id}, located at ({self.pos[0]}, {self.pos[1]})"

    def __repr__(self):
        return f"bot {self.id}"

    def step(self):
        self.navigation_table = copy.deepcopy(self.new_information)
        self.navigation_table.decay_qualities()
        self.move()
        self.update_trace()
        self.update_goal_vectors()
        self.update_behavior()
        self.update_orientation_based_on_state()

    def communicate(self, neighbors):
        self.new_information = copy.deepcopy(self.navigation_table)
        # strategy = BetterAgeStrategy()
        # strategy = WeightedAverageAgeStrategy()
        # strategy = QualityStrategy(1-abs(self.noise_mu))
        strategy = DecayingQualityStrategy(1-abs(self.noise_mu))
        for neighbor in neighbors:
            for location in Location:
                if strategy.should_combine(self.new_information.get_target(location), neighbor.get_nav_target(location)):
                    new_target = strategy.combine(self.new_information.get_target(location), neighbor.get_nav_target(location), neighbor.pos - self.pos)
                    self.new_information.replace_target(location, new_target)

    def get_nav_target(self, location):
        return self.navigation_table.get_target(location)

    def get_nav_location_age(self, location):
        return self.navigation_table.get_age(location)

    def get_location_vector(self, location):
        return self.navigation_table.get_location_vector(location)

    def knows_location(self, location):
        return self.navigation_table.is_location_known(location)

    def update_trace(self):
        self.trace.appendleft(self.pos[1])
        self.trace.appendleft(self.pos[0])

    def is_stationary(self):
        trace_length = len(self.trace) // 2
        tolerance = 3
        if trace_length < 4 * tolerance:  # If robot hasn't done 2*tolerance steps yet, consider it is moving
            return False
        trace_x = [self.trace[2*i] for i in range(min(trace_length, 4 * tolerance))]
        trace_y = [self.trace[2*i+1] for i in range(min(trace_length, 4 * tolerance))]
        max_x_gap = max(trace_x) - min(trace_x)
        max_y_gap = max(trace_y) - min(trace_y)

        return max_x_gap < tolerance * self.speed and max_y_gap < tolerance * self.speed

    def update_behavior(self):
        sensing_food = self.environment.senses_food(self)
        sensing_nest = self.environment.senses_nest(self)
        if sensing_food:
            self.set_food_vector()
            self.navigation_table.set_location_known(Location.FOOD, True)
            self.navigation_table.set_location_age(Location.FOOD, 0)
            self.navigation_table.reset_quality(Location.FOOD, 1-abs(self.noise_mu))
            # self.knows_food_location = True
            self.carries_food = True
        if sensing_nest:
            self.set_nest_vector()
            self.navigation_table.set_location_known(Location.NEST, True)
            self.navigation_table.set_location_age(Location.NEST, 0)
            self.navigation_table.reset_quality(Location.NEST, 1-abs(self.noise_mu))
            if self.carries_food:
                self.reward += 1
                # print(f"bot {self.id} rewarded, total reward={self.reward}, qualities={self.navigation_table.targets[Location.FOOD].quality, self.navigation_table.targets[Location.FOOD].decaying_quality }")
                self.carries_food = False

        if self.state == State.EXPLORING:
            if self.navigation_table.is_location_known(Location.FOOD) and not self.carries_food:
                self.state = State.SEEKING_FOOD
            if self.navigation_table.is_location_known(Location.NEST) and self.carries_food:
                self.state = State.SEEKING_NEST

        elif self.state == State.SEEKING_FOOD and sensing_food:
            if self.navigation_table.is_location_known(Location.NEST):
                self.state = State.SEEKING_NEST
            else:
                self.state = State.EXPLORING

        elif self.state == State.SEEKING_NEST and sensing_nest:
            if self.navigation_table.is_location_known(Location.FOOD):
                self.state = State.SEEKING_FOOD
            else:
                self.state = State.EXPLORING

        if self.is_stationary():
            if self.state == State.SEEKING_NEST:
                self.navigation_table.set_location_known(Location.NEST, False)
                self.state = State.EXPLORING
            elif self.state == State.SEEKING_FOOD:
                self.navigation_table.set_location_known(Location.FOOD, False)
                self.state = State.EXPLORING

    def update_orientation_based_on_state(self):
        turn_angle = 0
        if self.state == State.EXPLORING:
            turn_angle = self.get_levi_turn_angle()
        elif self.state == State.SEEKING_FOOD:
            turn_angle = get_orientation_from_vector(self.navigation_table.get_location_vector(Location.FOOD)) - self.orientation
        elif self.state == State.SEEKING_NEST:
            turn_angle = get_orientation_from_vector(self.navigation_table.get_location_vector(Location.NEST)) - self.orientation
        self.turn(turn_angle)

    def update_goal_vectors(self):
        self.navigation_table.update_from_movement(self.displacement)

    def set_food_vector(self):
        self.navigation_table.set_location_vector(Location.FOOD, self.environment.get_food_location() - self.pos)

    def set_nest_vector(self):
        self.navigation_table.set_location_vector(Location.NEST, self.environment.get_nest_location() - self.pos)

    def move(self):
        # Robot's will : calculates where it wants to end up and check if there are no border walls
        self.displacement = self.speed * np.array([cos(radians(self.orientation)), sin(radians(self.orientation))])
        collide_x, collide_y = self.environment.check_border_collision(self,
                                                                       self.pos[0] + self.displacement[0],
                                                                       self.pos[1] + self.displacement[1])
        # If border in front, flip orientation along wall axis
        if collide_x:
            self.flip_horizontally()
        if collide_y:
            self.flip_vertically()

        # Real movement, subject to noise, clamped to borders
        noise_amt = gauss(self.noise_mu, self.noise_sd)
        n_hor = noise_amt * self.speed
        n_vert = 0

        noise_rel = np.array([n_vert, n_hor])  # noise vector in robot's relative coordinates
        noise = rotation_matrix(self.orientation).dot(noise_rel)  # noise vector in absolute (x, y) coordinates
        real_displacement = (self.displacement + noise) * (self.speed / norm(self.displacement + noise))
        self.pos = self.clamp_to_map(self.pos + real_displacement)

    def turn(self, angle):
        noise_angle = gauss(self.noise_mu, self.noise_sd)
        # noise_angle = 2 * (random() - 0.5)
        self.rotate_vectors(noise_angle)
        self.orientation = (self.orientation + angle) % 360

    def rotate_vectors(self, angle):
        self.navigation_table.rotate_from_angle(angle)

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
            self.levi_counter = choices(range(1, self.max_levi_steps + 1), self.levi_weights)[0]

    def get_levi_turn_angle(self):
        angle = 0
        if self.levi_counter <= 1:
            angle = choices(self.thetas, self.crw_weights)[0]
        self.update_levi_counter()
        return angle

    def flip_horizontally(self):
        self.orientation = (180 - self.orientation) % 360
        self.displacement[0] = -self.displacement[0]

    def flip_vertically(self):
        self.orientation = (-self.orientation) % 360
        self.displacement[1] = -self.displacement[1]

    def draw(self, canvas):
        circle = canvas.create_oval(self.pos[0] - self.radius,
                                    self.pos[1] - self.radius,
                                    self.pos[0] + self.radius,
                                    self.pos[1] + self.radius,
                                    fill=self.colors[self.state])
        self.draw_comm_radius(canvas)
        self.draw_goal_vector(canvas)
        # self.draw_trace(canvas)

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
                                   self.pos[0] + self.navigation_table.get_location_vector(Location.FOOD)[0],
                                   self.pos[1] + self.navigation_table.get_location_vector(Location.FOOD)[1],
                                   arrow=LAST)
        arrow = canvas.create_line(self.pos[0],
                                   self.pos[1],
                                   self.pos[0] + self.navigation_table.get_location_vector(Location.NEST)[0],
                                   self.pos[1] + self.navigation_table.get_location_vector(Location.NEST)[1],
                                   arrow=LAST)

    def crw_pdf(self, thetas):
        res = []
        for t in thetas:
            num = (1 - self.rdwalk_factor ** 2)
            denom = 2 * pi * (1 + self.rdwalk_factor ** 2 - 2 * self.rdwalk_factor * cos(radians(t)))
            f = 1
            if denom != 0:
                f = num / denom
            res.append(f)
        return res

    def levi_pdf(self, max_steps):
        alpha = self.levi_factor
        pdf = [step ** (-alpha - 1) for step in range(1, max_steps + 1)]
        return pdf

    def sample_noise(self, noise_mu, noise_musd, noise_sd):
        mu = gauss(noise_mu, noise_musd)
        return mu, noise_sd

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
