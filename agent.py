from random import random, choices
from math import sin, cos, radians, pi, atan2
from tkinter import LAST
from enum import Enum
from collections import deque
import numpy as np


class State(Enum):
    EXPLORING = 1
    SEEKING_FOOD = 2
    SEEKING_NEST = 3


def norm(vector):
    return (sum(x ** 2 for x in vector)) ** 0.5


def get_orientation_from_vector(vector):
    angle = atan2(vector[1], vector[0])
    return (360 * angle / (2 * pi)) % 360


class Agent:
    thetas = np.arange(0, 360)
    weights = []
    colors = {State.EXPLORING: "blue", State.SEEKING_FOOD: "orange", State.SEEKING_NEST: "green"}

    def __init__(self, robot_id, x, y, speed, radius, rdwalk_factor, environment):
        self.id = robot_id
        self.pos = np.array([x, y]).astype('float64')
        self.speed = speed
        self.radius = radius
        self.rdwalk_factor = rdwalk_factor
        self.orientation = random() * 360  # 360 degree angle
        self.displacement = np.array([0, 0]).astype('float64')
        self.environment = environment
        self.state = State.EXPLORING
        self.reward = 0
        self.weights = self.crw_pdf(self.thetas)
        self.trace = deque(self.pos, maxlen=200)
        self.food_location_vector = np.array([0, 0]).astype('float64')
        self.nest_location_vector = np.array([0, 0]).astype('float64')
        self.knows_food_location = False
        self.knows_nest_location = False
        self.carries_food = False

    def __str__(self):
        return f"Bip boop, I am bot {self.id}, located at ({self.x}, {self.y})"

    def step(self):
        self.move()
        self.update_trace()
        self.update_goal_vectors()
        self.update_behavior()
        self.update_orientation_based_on_state()

    def update_trace(self):
        self.trace.appendleft(self.pos[1])
        self.trace.appendleft(self.pos[0])

    def update_behavior(self):
        sensing_food = self.environment.senses_food(self)
        sensing_nest = self.environment.senses_nest(self)
        if sensing_food:
            self.set_food_vector()
            self.knows_food_location = True
            self.carries_food = True
        if sensing_nest:
            self.set_nest_vector()
            self.knows_nest_location = True
            if self.carries_food:
                self.reward += 1
                self.carries_food = False

        if self.state == State.EXPLORING:
            if self.knows_food_location and sensing_nest:
                self.state = State.SEEKING_FOOD
            if self.knows_nest_location and sensing_food:
                self.state = State.SEEKING_NEST

        elif self.state == State.SEEKING_FOOD and sensing_food:
            if self.knows_nest_location:
                self.state = State.SEEKING_NEST
            else:
                self.state = State.EXPLORING

        elif self.state == State.SEEKING_NEST and sensing_nest:
            if self.knows_food_location:
                self.state = State.SEEKING_FOOD
            else:
                self.state = State.EXPLORING

    def update_orientation_based_on_state(self):

        if self.state == State.EXPLORING:
            self.turn_randomly()
        elif self.state == State.SEEKING_FOOD:
            self.orientation = get_orientation_from_vector(self.food_location_vector)
        elif self.state == State.SEEKING_NEST:
            self.orientation = get_orientation_from_vector(self.nest_location_vector)

    def update_goal_vectors(self):
        self.food_location_vector -= self.displacement
        self.nest_location_vector -= self.displacement

    def set_food_vector(self):
        self.food_location_vector = self.environment.get_food_location() - self.pos

    def set_nest_vector(self):
        self.nest_location_vector = self.environment.get_nest_location() - self.pos

    def move(self):
        self.displacement = self.speed * np.array([cos(radians(self.orientation)), sin(radians(self.orientation))])
        collide_x, collide_y = self.environment.check_border_collision(self,
                                                                       self.pos[0] + self.displacement[0],
                                                                       self.pos[1] + self.displacement[1])
        if collide_x:
            self.flip_horizontally()
        if collide_y:
            self.flip_vertically()
        self.pos += self.displacement

    def turn_randomly(self):
        angle = choices(self.thetas, self.weights)[0]
        self.orientation = (self.orientation + angle) % 360

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
        self.draw_goal_vector(canvas)
        self.draw_trace(canvas)

    def draw_trace(self, canvas):
        tail = canvas.create_line(*self.trace)

    def draw_goal_vector(self, canvas):
        goal_vector = np.array([0, 0])
        if self.state == State.SEEKING_FOOD:
            goal_vector = self.food_location_vector
        elif self.state == State.SEEKING_NEST:
            goal_vector = self.nest_location_vector
        arrow = canvas.create_line(self.pos[0], self.pos[1], self.pos[0] + goal_vector[0], self.pos[1] + goal_vector[1],
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
