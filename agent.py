from random import random, choices
from math import sin, cos, radians, pi
from tkinter import LAST


class Agent:
    thetas = range(0, 360)
    weights = []

    def __init__(self, robot_id, x, y, speed, radius, rdwalk_factor, environment):
        self.id = robot_id
        self.x = x
        self.y = y
        self.speed = speed
        self.radius = radius
        self.rdwalk_factor = rdwalk_factor
        self.orientation = random() * 360  # 360 degree angle
        self.dx = self.speed * cos(radians(self.orientation))
        self.dy = self.speed * sin(radians(self.orientation))
        self.environment = environment
        self.searching_food = True
        self.reward = 0
        self.weights = self.crw_pdf(self.thetas)
        goal_x, goal_y = self.environment.get_food_location()
        self.goal_vector = [goal_x - self.x, goal_y - self.y]

    def __str__(self):
        return f"Bip boop, I am bot {self.id}, located at ({self.x}, {self.y})"

    def step(self):
        self.move()
        self.goal_vector[0] -= self.dx # + (random()-0.5)
        self.goal_vector[1] -= self.dy # + (random()-0.5)
        self.turn()
        # self.add_orientation_noise() # odometry
        # self.orientation += (random()-0.5)*45
        # add random walk code
        sensing_food = self.environment.senses_food(self)
        sensing_nest = self.environment.senses_nest(self)
        if self.searching_food and sensing_food:
            self.searching_food = False
            goal_x, goal_y = self.environment.get_nest_location()
            self.goal_vector = [goal_x - self.x, goal_y - self.y]
        if not self.searching_food and sensing_nest:
            self.searching_food = True
            self.reward += 1
            goal_x, goal_y = self.environment.get_food_location()
            self.goal_vector = [goal_x - self.x, goal_y - self.y]

    def move(self):
        # self.orientation = (self.orientation + self.rdwalk_factor * (random()-0.5) * 360) % 360
        # self.dx = (1-self.rdwalk_factor) * self.speed * cos(radians(self.orientation)) + self.rdwalk_factor * self.speed * cos(radians(random()*360))
        # self.dy = (1-self.rdwalk_factor) * self.speed * sin(radians(self.orientation)) + self.rdwalk_factor * self.speed * sin(radians(random()*360))
        self.dx = self.speed * cos(radians(self.orientation))
        self.dy = self.speed * sin(radians(self.orientation))
        collide_x, collide_y = self.environment.check_border_collision(self, self.x + self.dx, self.y + self.dy)
        if collide_x:
            self.flip_horizontally()
        if collide_y:
            self.flip_vertically()
        self.x += self.dx
        self.y += self.dy

    def turn(self):
        angle = 0
        if self.rdwalk_factor != 1:
            angle = choices(self.thetas, self.weights)[0]
        self.orientation = (self.orientation + angle) % 360

    def flip_horizontally(self):
        self.orientation = (180 - self.orientation) % 360
        self.dx = -self.dx

    def flip_vertically(self):
        self.orientation = (-self.orientation) % 360
        self.dy = -self.dy

    def draw(self, canvas):
        if self.searching_food:
            circle = canvas.create_oval(self.x - self.radius,
                                        self.y - self.radius,
                                        self.x + self.radius,
                                        self.y + self.radius,
                                        fill="blue")
        else:
            circle = canvas.create_oval(self.x - self.radius,
                                        self.y - self.radius,
                                        self.x + self.radius,
                                        self.y + self.radius,
                                        fill="red")
        self.draw_goal_vector(canvas)

    def draw_goal_vector(self, canvas):
        arrow = canvas.create_line(self.x, self.y, self.x + self.goal_vector[0], self.y + self.goal_vector[1], arrow=LAST)

    def crw_pdf(self, thetas):
        res = []
        for t in thetas:
            num = (1 - self.rdwalk_factor ** 2)
            denom = 2 * pi * (1 + self.rdwalk_factor ** 2 - 2 * self.rdwalk_factor * cos(radians(t)))
            f = 1
            if denom != 0:
                f = num/denom
            res.append(f)
        print(res)
        return res
