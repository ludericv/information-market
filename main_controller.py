from environment import Environment
from view_controller import ViewController


class MainController:

    def __init__(self, config_file="config.txt"):
        self.parameters = dict()
        self.read_config(config_file)
        self.environment = Environment(width=self.parameters["WIDTH"],
                                       height=self.parameters["HEIGHT"],
                                       nb_robots=self.parameters["NB_ROBOTS"],
                                       robot_speed=self.parameters["ROBOT_SPEED"],
                                       robot_radius=self.parameters["ROBOT_RADIUS"],
                                       rdwalk_factor=self.parameters["RDWALK_FACTOR"],
                                       levi_factor=self.parameters["LEVI_FACTOR"],
                                       food_x=self.parameters["FOOD_X"],
                                       food_y=self.parameters["FOOD_Y"],
                                       food_radius=self.parameters["FOOD_RADIUS"],
                                       nest_x=self.parameters["NEST_X"],
                                       nest_y=self.parameters["NEST_Y"],
                                       nest_radius=self.parameters["NEST_RADIUS"],
                                       noise_mu=self.parameters["NOISE_MU"],
                                       noise_musd=self.parameters["NOISE_MUSD"],
                                       noise_sd=self.parameters["NOISE_SD"]
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
        float_params = {"RDWALK_FACTOR", "ROBOT_SPEED", "LEVI_FACTOR", "NOISE_MU", "NOISE_MUSD", "NOISE_SD"}
        if parameter in float_params:
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
