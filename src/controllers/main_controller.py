from helpers import random_walk

from model.environment import Environment


class Configuration:
    def __init__(self, config_file):
        self.parameters = dict()
        self.read_config(config_file)

    def read_config(self, config_file):
        with open(config_file, "r") as file:
            for line in file:
                args = line.strip().split("=")
                parameter = args[0]
                value = args[1]
                self.add_to_parameters(parameter, value)

    def add_to_parameters(self, parameter, value):
        float_params = {"RDWALK_FACTOR", "ROBOT_SPEED", "LEVI_FACTOR", "NOISE_MU", "NOISE_MUSD",
                        "NOISE_SD", "COMM_RADIUS", "INITIAL_REWARD", "FUEL_COST", "INFO_COST", "DEMAND", "MAX_PRICE"}
        if parameter in float_params:
            self.parameters[parameter] = float(value)
        else:
            self.parameters[parameter] = int(value)


class MainController:

    def __init__(self, config: Configuration):
        self.config = config
        random_walk.set_parameters(random_walk_factor=self.config.parameters["RDWALK_FACTOR"],
                                   levi_factor=self.config.parameters["LEVI_FACTOR"]
                                   # , max_levi_steps=self.config.parameters["SIMULATION_STEPS"]
                                   )
        self.environment = Environment(width=self.config.parameters["WIDTH"],
                                       height=self.config.parameters["HEIGHT"],
                                       nb_robots=self.config.parameters["NB_ROBOTS"],
                                       nb_honest=self.config.parameters["NB_HONEST"],
                                       robot_speed=self.config.parameters["ROBOT_SPEED"],
                                       comm_radius=self.config.parameters["COMM_RADIUS"],
                                       robot_radius=self.config.parameters["ROBOT_RADIUS"],
                                       rdwalk_factor=self.config.parameters["RDWALK_FACTOR"],
                                       levi_factor=self.config.parameters["LEVI_FACTOR"],
                                       food_x=self.config.parameters["FOOD_X"],
                                       food_y=self.config.parameters["FOOD_Y"],
                                       food_radius=self.config.parameters["FOOD_RADIUS"],
                                       nest_x=self.config.parameters["NEST_X"],
                                       nest_y=self.config.parameters["NEST_Y"],
                                       nest_radius=self.config.parameters["NEST_RADIUS"],
                                       noise_mu=self.config.parameters["NOISE_MU"],
                                       noise_musd=self.config.parameters["NOISE_MUSD"],
                                       noise_sd=self.config.parameters["NOISE_SD"],
                                       initial_reward=self.config.parameters["INITIAL_REWARD"],
                                       fuel_cost=self.config.parameters["FUEL_COST"],
                                       info_cost=self.config.parameters["INFO_COST"],
                                       demand=self.config.parameters["DEMAND"],
                                       max_price=self.config.parameters["MAX_PRICE"],
                                       robot_comm_cooldown=self.config.parameters["COMM_COOLDOWN"],
                                       robot_comm_stop_time=self.config.parameters["COMM_STOPTIME"]
                                       )
        self.tick = 0

    def step(self):
        if self.tick < self.config.parameters["SIMULATION_STEPS"]:
            self.tick += 1
            self.environment.step()

    def start_simulation(self):
        # now = time.time()
        for step_nb in range(self.config.parameters["SIMULATION_STEPS"]):
            self.step()
        # print(f"Time taken for {self.config.parameters['SIMULATION_STEPS']} steps: {time.time()-now}")

    def get_sorted_reward_stats(self):
        sorted_bots = sorted([bot for bot in self.environment.population], key=lambda bot: abs(bot.noise_mu))
        res = ""
        for bot in sorted_bots:
            res += str(bot.reward()) + ","
        res = res[:-1]  # remove last comma
        res += "\n"
        return res

    def get_reward_stats(self):
        res = ""
        for bot in self.environment.population:
            res += str(bot.reward()) + ","
        res = res[:-1]  # remove last comma
        res += "\n"
        return res

    def get_items_collected_stats(self):
        res = ""
        for bot in self.environment.population:
            res += str(bot.items_collected) + ","
        res = res[:-1]  # remove last comma
        res += "\n"
        return res

    def get_robot_at(self, x, y):
        return self.environment.get_robot_at(x, y)
