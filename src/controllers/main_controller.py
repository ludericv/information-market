import json

from helpers import random_walk

from model.environment import Environment


class Configuration:
    def __init__(self, config_file):
        self._parameters = self.read_config(config_file)

    @staticmethod
    def read_config(config_file):
        with open(config_file, "r") as file:
            return json.load(file)

    def save(self, save_path):
        json_object = json.dumps(self._parameters, indent=2)
        with open(save_path, 'w') as file:
            file.write(json_object)

    def value_of(self, parameter):
        return self._parameters[parameter]

    def set(self, parameter, value):
        self._parameters[parameter] = value


class MainController:

    def __init__(self, config: Configuration):
        self.config = config
        random_walk.set_parameters(**self.config.value_of('random_walk'),
                                   max_levi_steps=self.config.value_of("simulation_steps")+1
                                   )
        self.environment = Environment(width=self.config.value_of("width"),
                                       height=self.config.value_of("height"),
                                       agent_params=self.config.value_of("agent"),
                                       behavior_params=self.config.value_of("behaviors"),
                                       food=self.config.value_of("food"),
                                       nest=self.config.value_of("nest"),
                                       payment_system_params=config.value_of("payment_system"),
                                       market_params=config.value_of("market")
                                       )
        self.tick = 0
        self.rewards_evolution = ""

    def step(self):
        if self.config.value_of("data_collection")['precision_recording'] and \
                self.tick % self.config.value_of("data_collection")['precision_recording_interval'] == 0:
            self.rewards_evolution += f"{self.tick},{self.get_reward_stats()}"
        if self.tick < self.config.value_of("simulation_steps"):
            self.tick += 1
            self.environment.step()

    def start_simulation(self):
        for step_nb in range(self.config.value_of("simulation_steps")):
            self.step()

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

    def get_rewards_evolution(self):
        return self.rewards_evolution
