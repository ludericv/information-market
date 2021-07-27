from environment import Environment
from view_controller import ViewController


class MainController:

    def __init__(self, config_file="config.txt"):
        self.parameters = dict()
        self.read_config(config_file)
        self.environment = Environment(self.parameters["WIDTH"],
                                       self.parameters["HEIGHT"],
                                       self.parameters["NB_ROBOTS"],
                                       self.parameters["ROBOT_SPEED"],
                                       self.parameters["ROBOT_RADIUS"],
                                       self.parameters["RDWALK_FACTOR"],
                                       self.parameters["FOOD_X"],
                                       self.parameters["FOOD_Y"],
                                       self.parameters["FOOD_RADIUS"],
                                       self.parameters["NEST_X"],
                                       self.parameters["NEST_Y"],
                                       self.parameters["NEST_RADIUS"]
                                       )
        self.tick = 0
        if self.parameters["VISUALIZE"] != 0:
            self.view_controller = ViewController(self,
                                                  self.parameters["WIDTH"],
                                                  self.parameters["HEIGHT"],
                                                  self.parameters["FPS"])
        else:
            self.start_simulation()

    def read_config(self, config_file):
        with open(config_file, "r") as file:
            for line in file:
                args = line.strip().split("=")
                parameter = args[0]
                value = args[1]
                self.add_to_parameters(parameter, value)

    def add_to_parameters(self, parameter, value):
        if parameter == "RDWALK_FACTOR" or parameter == "ROBOT_SPEED":
            self.parameters[parameter] = float(value)
        else:
            self.parameters[parameter] = int(value)

    def step(self):
        if self.tick < self.parameters["SIMULATION_STEPS"]:
            self.tick += 1
            self.environment.step()

    def start_simulation(self):
        for step_nb in range(self.parameters["SIMULATION_STEPS"]):
            self.step()
            print(self.environment.population[0])
