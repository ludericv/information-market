from random import random
from math import sin, cos, radians


class Agent:
    def __init__(self, id, x, y, speed, radius, environment):
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed
        self.radius = radius
        self.orientation = random() * 360  # 360 degree angle
        self.environment = environment

    def __str__(self):
        return f"Bip boop, I am bot {self.id}, located at ({self.x}, {self.y}), with orientation {round(self.orientation, 2)}"

    def step(self):
        self.move()
        self.add_orientation_noise()

    def move(self):
        dx = self.speed * cos(radians(self.orientation))
        dy = self.speed * sin(radians(self.orientation))
        self.environment.check_border_collision(self, self.x + dx, self.y + dy)
        self.x += self.speed * cos(radians(self.orientation))
        self.y += self.speed * sin(radians(self.orientation))

    def flip_horizontally(self):
        self.orientation = (180 - self.orientation) % 360

    def flip_vertically(self):
        self.orientation = (-self.orientation) % 360

    def add_orientation_noise(self):
        self.orientation = (self.orientation + (random() - 0.5) * 25) % 360

    def draw(self, canvas):
        circle = canvas.create_oval(self.x - self.radius,
                                    self.y - self.radius,
                                    self.x + self.radius,
                                    self.y + self.radius,
                                    fill="blue")
