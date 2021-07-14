from agent import Agent
from random import randint


class Environment:

    def __init__(self, config_file="config.txt"):
        self.parameters = dict()
        self.read_config(config_file)
        self.population = list()
        self.create_robots()

    def read_config(self, config_file):
        with open(config_file, "r") as file:
            for line in file:
                args = line.strip().split("=")
                parameter = args[0]
                value = args[1]
                self.parameters[parameter] = int(value)

    def create_robots(self):
        for id in range(self.parameters["NB_ROBOTS"]):
            robot = Agent(id=id,
                          x=randint(0, self.parameters["WIDTH"]-1),
                          y=randint(0, self.parameters["HEIGHT"]-1),
                          speed=self.parameters["ROBOT_SPEED"],
                          environment=self)
            self.population.append(robot)

    def start_simulation(self):
        for tick in range(self.parameters["SIMULATION_STEPS"]):
            for robot in self.population:
                robot.step()
            print(self.population[0])

    def check_border_collision(self, robot, newX, newY):
        if newX >= self.parameters["WIDTH"] or newX < 0:
            robot.flip_horizontally()

        if newY >= self.parameters["HEIGHT"] or newY < 0:
            robot.flip_vertically()

