from math import cos, sin

from agent import Agent
from utils import norm
from random import randint
from copy import deepcopy, copy
import numpy as np


def distance_between(robot1, robot2):
    return norm(robot2.pos - robot1.pos)


class Environment:

    def __init__(self, width=500, height=500, nb_robots=30, robot_speed=3, robot_radius=5, comm_radius=25,
                 rdwalk_factor=0,
                 levi_factor=2, food_x=0, food_y=0, food_radius=25, nest_x=500, nest_y=500, nest_radius=25, noise_mu=0,
                 noise_musd=1, noise_sd=0.1):
        self.population = list()
        self.width = width
        self.height = height
        self.nb_robots = nb_robots
        self.robot_speed = robot_speed
        self.robot_radius = robot_radius
        self.robot_communication_radius = comm_radius
        self.rdwalk_factor = rdwalk_factor
        self.levi_factor = levi_factor
        self.food = (food_x, food_y, food_radius)
        self.nest = (nest_x, nest_y, nest_radius)
        self.noise_mu = noise_mu
        self.noise_musd = noise_musd
        self.noise_sd = noise_sd
        self.create_robots()
        self.best_bot_id = self.get_best_bot_id()

    def step(self):
        # compute neighbors
        pop_size = len(self.population)
        neighbors_table = [[] for i in range(pop_size)]
        for id1 in range(pop_size):
            for id2 in range(id1 + 1, pop_size):
                if distance_between(self.population[id1], self.population[id2]) < self.robot_communication_radius:
                    neighbors_table[id1].append(self.population[id2])
                    neighbors_table[id2].append(self.population[id1])

        # Iterate over robots
        # 1. Negotiation/communication
        for robot in self.population:
            # print(neighbors_table[robot.id])
            robot.communicate(neighbors_table[robot.id])

        # 2. Move
        for robot in self.population:
            robot.step()

    def create_robots(self):
        for robot_id in range(self.nb_robots):
            robot = Agent(robot_id=robot_id,
                          x=randint(self.robot_radius, self.width - 1 - self.robot_radius),
                          y=randint(self.robot_radius, self.height - 1 - self.robot_radius),
                          speed=self.robot_speed,
                          radius=self.robot_radius,
                          rdwalk_factor=self.rdwalk_factor,
                          levi_factor=self.levi_factor,
                          noise_mu=self.noise_mu,
                          noise_musd=self.noise_musd,
                          noise_sd=self.noise_sd,
                          environment=self)
            self.population.append(robot)

    def get_sensors(self, robot):
        orientation = robot.orientation
        speed = robot.speed
        sensors = {"FOOD": self.senses_food(robot),
                   "NEST": self.senses_nest(robot),
                   "FRONT": any(self.check_border_collision(robot, robot.pos[0] + speed * cos(orientation),
                                                            robot.pos[1] + speed * sin(orientation))),
                   "RIGHT": any(self.check_border_collision(robot, robot.pos[0] + speed * cos((orientation - 90) % 360),
                                                            robot.pos[1] + speed * sin((orientation - 90) % 360))),
                   "BACK": any(self.check_border_collision(robot, robot.pos[0] + speed * cos((orientation + 180) % 360),
                                                           robot.pos[1] + speed * sin((orientation + 180) % 360))),
                   "LEFT": any(self.check_border_collision(robot, robot.pos[0] + speed * cos((orientation + 90) % 360),
                                                           robot.pos[1] + speed * sin((orientation + 90) % 360))),
                   }
        return sensors

    def check_border_collision(self, robot, new_x, new_y):
        collide_x = False
        collide_y = False
        if new_x + robot.radius >= self.width or new_x - robot.radius < 0:
            collide_x = True

        if new_y + robot.radius >= self.height or new_y - robot.radius < 0:
            collide_y = True

        return collide_x, collide_y

    def senses_food(self, robot):
        dist_from_center = (robot.pos[0] - self.food[0]) ** 2 + (robot.pos[1] - self.food[1]) ** 2
        return dist_from_center < self.food[2] ** 2

    def senses_nest(self, robot):
        dist_from_center = (robot.pos[0] - self.nest[0]) ** 2 + (robot.pos[1] - self.nest[1]) ** 2
        return dist_from_center < self.nest[2] ** 2

    def get_food_location(self):
        return np.array([self.food[0], self.food[1]])

    def get_nest_location(self):
        return np.array([self.nest[0], self.nest[1]])

    def draw(self, canvas):
        self.draw_zones(canvas)
        for robot in self.population:
            robot.draw(canvas)
        self.draw_best_bot(canvas)

    def draw_zones(self, canvas):
        food_circle = canvas.create_oval(self.food[0] - self.food[2],
                                         self.food[1] - self.food[2],
                                         self.food[0] + self.food[2],
                                         self.food[1] + self.food[2],
                                         fill="green")
        nest_circle = canvas.create_oval(self.nest[0] - self.nest[2],
                                         self.nest[1] - self.nest[2],
                                         self.nest[0] + self.nest[2],
                                         self.nest[1] + self.nest[2],
                                         fill="orange")

    def get_best_bot_id(self):
        best_bot_id = 0
        for bot in self.population:
            if 1 - abs(bot.noise_mu) > 1 - abs(self.population[best_bot_id].noise_mu):
                best_bot_id = bot.id
        return best_bot_id

    def draw_best_bot(self, canvas):
        circle = canvas.create_oval(self.population[self.best_bot_id].pos[0] - 4,
                                    self.population[self.best_bot_id].pos[1] - 4,
                                    self.population[self.best_bot_id].pos[0] + 4,
                                    self.population[self.best_bot_id].pos[1] + 4,
                                    fill="red")

    def get_robot_at(self, x, y):
        selected = None
        for bot in self.population:
            if norm(bot.pos - np.array([x, y]).astype('float64')) < self.robot_radius:
                selected = bot
                break

        return selected
