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
                                       self.parameters["ROBOT_RADIUS"])
        if self.parameters["VISUALIZE"] != 0:
            self.view_controller = ViewController(self.parameters["WIDTH"],
                                                  self.parameters["HEIGHT"],
                                                  self.parameters["FPS"])

    def read_config(self, config_file):
        with open(config_file, "r") as file:
            for line in file:
                args = line.strip().split("=")
                parameter = args[0]
                value = args[1]
                self.parameters[parameter] = int(value)

    def start_simulation(self):
        for tick in range(self.parameters["SIMULATION_STEPS"]):
            self.environment.step()
            if self.parameters["VISUALIZE"] != 0:
                self.view_controller.update(self.environment, tick)
            else:
                print(self.environment.population[0])
