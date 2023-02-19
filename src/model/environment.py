from math import cos, sin, radians
from PIL import ImageTk
from model.agent import Agent
from model.market import market_factory
from model.navigation import Location
from helpers.utils import norm, distance_between
from random import randint, random
import numpy as np

from model.payment import PaymentDB


class Environment:

    def __init__(self, width, height, agent_params, behavior_params, food, nest, payment_system_params, market_params, clock):
        self.population = list()
        self.width = width
        self.height = height
        self.clock = clock
        self.food = (food['x'], food['y'], food['radius'])
        self.nest = (nest['x'], nest['y'], nest['radius'])
        self.locations = {Location.FOOD: self.food, Location.NEST: self.nest}
        self.foraging_spawns = self.create_spawn_dicts()
        self.create_robots(agent_params, behavior_params)
        self.best_bot_id = self.get_best_bot_id()
        self.payment_database = PaymentDB([bot.id for bot in self.population], payment_system_params)

        self.market = market_factory(market_params)
        self.img = None
        self.timestep = 0

    def load_images(self):
        self.img = ImageTk.PhotoImage(file="../assets/strawberry.png")

    def step(self):
        # compute neighbors
        pop_size = len(self.population)
        neighbors_table = [[] for i in range(pop_size)]
        for id1 in range(pop_size):
            for id2 in range(id1 + 1, pop_size):
                if distance_between(self.population[id1], self.population[id2]) < self.population[id1].communication_radius:
                    neighbors_table[id1].append(self.population[id2])
                    neighbors_table[id2].append(self.population[id1])

        # Iterate over robots
        # 1. Negotiation/communication
        for robot in self.population:
            robot.communicate(neighbors_table[robot.id])

        # 2. Move
        for robot in self.population:
            self.check_locations(robot)
            robot.step()

        self.market.step()

    def create_robots(self, agent_params, behavior_params):
        robot_id = 0
        for behavior_params in behavior_params:
            for _ in range(behavior_params['population_size']):
                robot = Agent(robot_id=robot_id,
                              x=randint(agent_params['radius'], self.width - 1 - agent_params['radius']),
                              y=randint(agent_params['radius'], self.height - 1 - agent_params['radius']),
                              environment=self,
                              behavior_params=behavior_params,
                              clock=self.clock,
                              **agent_params)
                robot_id += 1
                self.population.append(robot)

    def get_sensors(self, robot):
        orientation = robot.orientation
        speed = robot.speed()
        sensors = {Location.FOOD: self.senses(robot, Location.FOOD),
                   Location.NEST: self.senses(robot, Location.NEST),
                   "FRONT": any(self.check_border_collision(robot, robot.pos[0] + speed * cos(radians(orientation)),
                                                            robot.pos[1] + speed * sin(radians(orientation)))),
                   "RIGHT": any(
                       self.check_border_collision(robot, robot.pos[0] + speed * cos(radians((orientation - 90) % 360)),
                                                   robot.pos[1] + speed * sin(radians((orientation - 90) % 360)))),
                   "BACK": any(self.check_border_collision(robot, robot.pos[0] + speed * cos(
                       radians((orientation + 180) % 360)),
                                                           robot.pos[1] + speed * sin(
                                                               radians((orientation + 180) % 360)))),
                   "LEFT": any(
                       self.check_border_collision(robot, robot.pos[0] + speed * cos(radians((orientation + 90) % 360)),
                                                   robot.pos[1] + speed * sin(radians((orientation + 90) % 360)))),
                   }
        return sensors

    def check_border_collision(self, robot, new_x, new_y):
        collide_x = False
        collide_y = False
        if new_x + robot._radius >= self.width or new_x - robot._radius < 0:
            collide_x = True

        if new_y + robot._radius >= self.height or new_y - robot._radius < 0:
            collide_y = True

        return collide_x, collide_y

    def senses(self, robot, location):
        dist_vector = robot.pos - np.array([self.locations[location][0], self.locations[location][1]])
        dist_from_center = np.sqrt(dist_vector.dot(dist_vector))
        return dist_from_center < self.locations[location][2]

    def is_on_top_of_spawn(self, robot, location):
        dist_vector = robot.pos - self.foraging_spawns[location].get(robot.id)
        return np.sqrt(dist_vector.dot(dist_vector)) < robot._radius

    def get_location(self, location, agent):
        if agent.id in self.foraging_spawns[location]:
            return self.foraging_spawns[location][agent.id]
        else:
            return np.array([self.locations[location][0], self.locations[location][1]])

    def draw(self, canvas):
        self.draw_zones(canvas)
        self.draw_strawberries(canvas)
        for robot in self.population:
            robot.draw(canvas)
        # self.draw_best_bot(canvas)

    def draw_market_stats(self, stats_canvas):
        margin = 15
        width = stats_canvas.winfo_width() - 2 * margin
        height = 20
        stats_canvas.create_rectangle(margin, 50, margin + width, 50 + height, fill="light green", outline="")
        target_demand = self.market.demand
        max_theoretical_supply = self.market.demand/self.demand
        demand_pos_x = width*target_demand/max_theoretical_supply
        supply_pos_x = width*self.market.get_supply()/max_theoretical_supply
        supply_bar_width = 2
        stats_canvas.create_rectangle(margin + demand_pos_x, 50, margin + width, 50 + height, fill="salmon", outline="")
        stats_canvas.create_rectangle(margin + supply_pos_x - supply_bar_width/2, 48, margin + supply_pos_x + supply_bar_width/2, 52 + height, fill="gray45", outline="")
        stats_canvas.create_text(margin + supply_pos_x - 5, 50 + height + 5, fill="gray45", text=f"{round(self.market.get_supply())}", anchor="nw", font="Arial 10")

    def draw_zones(self, canvas):
        food_circle = canvas.create_oval(self.food[0] - self.food[2],
                                         self.food[1] - self.food[2],
                                         self.food[0] + self.food[2],
                                         self.food[1] + self.food[2],
                                         fill="green",
                                         outline="")
        nest_circle = canvas.create_oval(self.nest[0] - self.nest[2],
                                         self.nest[1] - self.nest[2],
                                         self.nest[0] + self.nest[2],
                                         self.nest[1] + self.nest[2],
                                         fill="orange",
                                         outline="")

    def get_best_bot_id(self):
        best_bot_id = 0
        for bot in self.population:
            if 1 - abs(bot.noise_mu) > 1 - abs(self.population[best_bot_id].noise_mu):
                best_bot_id = bot.id
        return best_bot_id

    def draw_strawberries(self, canvas):
        for bot_id, pos in self.foraging_spawns[Location.FOOD].items():
            canvas.create_image(pos[0] - 8, pos[1] - 8, image=self.img, anchor='nw')

        # for bot_id, pos in self.foraging_spawns[Location.NEST].items():
        #     canvas.create_image(pos[0] - 8, pos[1] - 8, image=self.img, anchor='nw')

    def draw_best_bot(self, canvas):
        circle = canvas.create_oval(self.population[self.best_bot_id].pos[0] - 4,
                                    self.population[self.best_bot_id].pos[1] - 4,
                                    self.population[self.best_bot_id].pos[0] + 4,
                                    self.population[self.best_bot_id].pos[1] + 4,
                                    fill="red")

    def get_robot_at(self, x, y):
        selected = None
        for bot in self.population:
            if norm(bot.pos - np.array([x, y]).astype('float64')) < bot.radius():
                selected = bot
                break

        return selected

    @staticmethod
    def create_spawn_dicts():
        d = dict()
        for location in Location:
            d[location] = dict()
        return d

    def check_locations(self, robot):
        if robot.carries_food():
            if self.senses(robot, Location.NEST):
                # Spawn deposit location if needed
                if robot.id not in self.foraging_spawns[Location.NEST]:
                    self.add_spawn(Location.NEST, robot)
                # Check if robot can deposit food
                if self.is_on_top_of_spawn(robot, Location.NEST):
                    self.deposit_food(robot)
        else:
            if self.senses(robot, Location.FOOD):
                # Spawn food if needed
                if robot.id not in self.foraging_spawns[Location.FOOD]:
                    self.add_spawn(Location.FOOD, robot)
                # Check if robot can pickup food
                if self.is_on_top_of_spawn(robot, Location.FOOD):
                    self.pickup_food(robot)

    def add_spawn(self, location, robot):
        rand_angle, rand_rad = random() * 360, np.sqrt(random()) * self.locations[location][2]
        pos_in_circle = rand_rad * np.array([cos(radians(rand_angle)), sin(radians(rand_angle))])
        self.foraging_spawns[location][robot.id] = np.array([self.locations[location][0],
                                                             self.locations[location][1]]) + pos_in_circle

    def deposit_food(self, robot):
        robot.drop_food()
        self.foraging_spawns[Location.NEST].pop(robot.id)

        reward = self.market.sell_strawberry(robot.id)

        self.payment_database.pay_reward(robot.id, reward=reward)
        self.payment_database.pay_creditors(robot.id, total_reward=reward)

    def pickup_food(self, robot):
        robot.pickup_food()
        self.foraging_spawns[Location.FOOD].pop(robot.id)
