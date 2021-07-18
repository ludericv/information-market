from agent import Agent
from random import randint


class Environment:

    def __init__(self, width=500, height=500, nb_robots=30, robot_speed=3, robot_radius=5):
        self.population = list()
        self.width = width
        self.height = height
        self.nb_robots = nb_robots
        self.robot_speed = robot_speed
        self.robot_radius = robot_radius
        self.create_robots()

    def step(self):
        for robot in self.population:
            robot.step()

    def create_robots(self):
        for id in range(self.nb_robots):
            robot = Agent(id=id,
                          x=randint(self.robot_radius, self.width-1-self.robot_radius),
                          y=randint(self.robot_radius, self.height-1-self.robot_radius),
                          speed=self.robot_speed,
                          radius=self.robot_radius,
                          environment=self)
            self.population.append(robot)

    def check_border_collision(self, robot, newX, newY):
        if newX+robot.radius >= self.width or newX-robot.radius < 0:
            robot.flip_horizontally()

        if newY+robot.radius >= self.height or newY-robot.radius < 0:
            robot.flip_vertically()

    def get_robot_positions(self):
        xdata = []
        ydata = []
        for robot in self.population:
            xdata.append(robot.x)
            ydata.append(robot.y)
        return xdata, ydata

    def draw(self, canvas):
        for robot in self.population:
            robot.draw(canvas)

