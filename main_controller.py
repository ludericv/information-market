import time
import random_walk

from environment import Environment
from view_controller import ViewController


class MainController:

    def __init__(self, config_file="config.txt"):
        self.parameters = dict()
        self.read_config(config_file)
        random_walk.set_parameters(random_walk_factor=self.parameters["RDWALK_FACTOR"],
                                   levi_factor=self.parameters["LEVI_FACTOR"])
        self.environment = Environment(width=self.parameters["WIDTH"],
                                       height=self.parameters["HEIGHT"],
                                       nb_robots=self.parameters["NB_ROBOTS"],
                                       robot_speed=self.parameters["ROBOT_SPEED"],
                                       comm_radius=self.parameters["COMM_RADIUS"],
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
                                       noise_sd=self.parameters["NOISE_SD"],
                                       initial_reward=self.parameters["INITIAL_REWARD"],
                                       fuel_cost=self.parameters["FUEL_COST"],
                                       info_cost=self.parameters["INFO_COST"]
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
        float_params = {"RDWALK_FACTOR", "ROBOT_SPEED", "LEVI_FACTOR", "NOISE_MU", "NOISE_MUSD",
                        "NOISE_SD", "COMM_RADIUS", "INITIAL_REWARD", "FUEL_COST", "INFO_COST"}
        if parameter in float_params:
            self.parameters[parameter] = float(value)
        else:
            self.parameters[parameter] = int(value)

    def step(self):
        if self.tick < self.parameters["SIMULATION_STEPS"]:
            self.tick += 1
            self.environment.step()

    def start_simulation(self):
        now = time.time()
        for step_nb in range(self.parameters["SIMULATION_STEPS"]):
            self.step()
        # print(f"Time taken for {self.parameters['SIMULATION_STEPS']} steps: {time.time()-now}")

    def get_reward_stats(self):
        res = ""
        for bot in self.environment.population:
            res += str(bot._reward) + ","
        res = res[:-1] # remove last comma
        res += "\n"
        return res

    def get_robot_at(self, x, y):
        return self.environment.get_robot_at(x, y)
