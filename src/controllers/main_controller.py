from helpers import random_walk

from model.environment import Environment


class Configuration:
    def __init__(self, config_file):
        self._parameters = dict()
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
            self._parameters[parameter] = float(value)
        else:
            self._parameters[parameter] = int(value)

    def value_of(self, parameter):
        return self._parameters[parameter]


class MainController:

    def __init__(self, config: Configuration):
        self.config = config
        random_walk.set_parameters(random_walk_factor=self.config.value_of("RDWALK_FACTOR"),
                                   levi_factor=self.config.value_of("LEVI_FACTOR")
                                   # , max_levi_steps=self.config.parameters["SIMULATION_STEPS"]
                                   )
        self.environment = Environment(width=self.config.value_of("WIDTH"),
                                       height=self.config.value_of("HEIGHT"),
                                       nb_robots=self.config.value_of("NB_ROBOTS"),
                                       nb_honest=self.config.value_of("NB_HONEST"),
                                       robot_speed=self.config.value_of("ROBOT_SPEED"),
                                       comm_radius=self.config.value_of("COMM_RADIUS"),
                                       robot_radius=self.config.value_of("ROBOT_RADIUS"),
                                       rdwalk_factor=self.config.value_of("RDWALK_FACTOR"),
                                       levi_factor=self.config.value_of("LEVI_FACTOR"),
                                       food_x=self.config.value_of("FOOD_X"),
                                       food_y=self.config.value_of("FOOD_Y"),
                                       food_radius=self.config.value_of("FOOD_RADIUS"),
                                       nest_x=self.config.value_of("NEST_X"),
                                       nest_y=self.config.value_of("NEST_Y"),
                                       nest_radius=self.config.value_of("NEST_RADIUS"),
                                       noise_mu=self.config.value_of("NOISE_MU"),
                                       noise_musd=self.config.value_of("NOISE_MUSD"),
                                       noise_sd=self.config.value_of("NOISE_SD"),
                                       initial_reward=self.config.value_of("INITIAL_REWARD"),
                                       fuel_cost=self.config.value_of("FUEL_COST"),
                                       info_cost=self.config.value_of("INFO_COST"),
                                       demand=self.config.value_of("DEMAND"),
                                       max_price=self.config.value_of("MAX_PRICE"),
                                       robot_comm_cooldown=self.config.value_of("COMM_COOLDOWN"),
                                       robot_comm_stop_time=self.config.value_of("COMM_STOPTIME")
                                       )
        self.tick = 0

    def step(self):
        if self.tick < self.config.value_of("SIMULATION_STEPS"):
            self.tick += 1
            self.environment.step()

    def start_simulation(self):
        # now = time.time()
        for step_nb in range(self.config.value_of("SIMULATION_STEPS")):
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

    def get_drift_stats(self):
        res = ""
        for bot in self.environment.population:
            res += str(bot.noise_mu) + ","
        res = res[:-1]  # remove last comma
        res += "\n"
        return res

    def get_robot_at(self, x, y):
        return self.environment.get_robot_at(x, y)
