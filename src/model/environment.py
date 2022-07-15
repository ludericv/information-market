from math import cos, sin, radians
from PIL import ImageTk
from model.agent import Agent
from model.behavior import SaboteurBehavior, CarefulBehavior, SmartBehavior, HonestBehavior, GreedyBehavior, \
    FreeRiderBehavior, SmartboteurBehavior, SmartGreedyBehavior
from model.market import Market, RoundTripPriceMarket, FixedPriceMarket
from model.navigation import Location
from helpers.utils import norm, distance_between
from random import randint, random
import numpy as np

from model.payment import PaymentDB


class Environment:

    def __init__(self, width=500, height=500, nb_robots=30, nb_honest=29, robot_speed=3, robot_radius=5, comm_radius=25,
                 rdwalk_factor=0,
                 levi_factor=2, food_x=0, food_y=0, food_radius=25, nest_x=500, nest_y=500, nest_radius=25, noise_mu=0,
                 noise_musd=1, noise_sd=0.1, initial_reward=3, fuel_cost=0.001, info_cost=0.01, demand=15, max_price=1,
                 robot_comm_cooldown=10, robot_comm_stop_time=5):
        self.population = list()
        self.width = width
        self.height = height
        self.nb_robots = nb_robots
        self.nb_honest = nb_honest
        self.robot_speed = robot_speed
        self.robot_radius = robot_radius
        self.robot_communication_radius = comm_radius
        self.robot_comm_cooldown=robot_comm_cooldown
        self.robot_comm_stop_time=robot_comm_stop_time
        self.rdwalk_factor = rdwalk_factor
        self.levi_factor = levi_factor
        self.food = (food_x, food_y, food_radius)
        self.nest = (nest_x, nest_y, nest_radius)
        self.locations = {Location.FOOD: self.food, Location.NEST: self.nest}
        self.noise_mu = noise_mu
        self.noise_musd = noise_musd
        self.noise_sd = noise_sd
        self.initial_reward = initial_reward
        self.fuel_cost = fuel_cost
        self.info_cost = info_cost
        self.foraging_spawns = self.create_spawn_dicts()
        self.create_robots()
        self.best_bot_id = self.get_best_bot_id()
        self.payment_database = PaymentDB([bot.id for bot in self.population], initial_reward, info_cost)

        # demand in items that "perfect" robots would collect per step
        self.demand = demand
        demand_converted = demand*nb_robots*robot_speed/(2*norm([food_x-nest_x, food_y-nest_y]))
        # self.market = RoundTripPriceMarket(2*norm([food_x-nest_x, food_y-nest_y])/robot_speed, max_price)
        self.market = FixedPriceMarket(max_price)
        self.img = None
        self.timestep = 0

    def load_images(self):
        self.img = ImageTk.PhotoImage(file="../assets/strawberry.png")

    def step(self):
        # compute neighbors
        self.timestep += 1
        pop_size = len(self.population)
        neighbors_table = [[] for i in range(pop_size)]
        for id1 in range(pop_size):
            for id2 in range(id1 + 1, pop_size):
                if distance_between(self.population[id1], self.population[id2]) < self.robot_communication_radius:
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

        self.payment_database.step()
        self.market.step()

    def create_robots(self):
        for robot_id in range(self.nb_honest):
            robot = Agent(robot_id=robot_id,
                          x=randint(self.robot_radius, self.width - 1 - self.robot_radius),
                          y=randint(self.robot_radius, self.height - 1 - self.robot_radius),
                          speed=self.robot_speed,
                          radius=self.robot_radius,
                          noise_mu=self.noise_mu,
                          noise_musd=self.noise_musd,
                          noise_sd=self.noise_sd,
                          initial_reward=self.initial_reward,
                          fuel_cost=self.fuel_cost,
                          info_cost=self.info_cost,
                          behavior=HonestBehavior(),  # Line that changes
                          environment=self,
                          communication_cooldown=self.robot_comm_cooldown,
                          communication_stop_time=self.robot_comm_stop_time)
            self.population.append(robot)
        for robot_id in range(self.nb_honest, self.nb_robots):
            robot = Agent(robot_id=robot_id,
                          x=randint(self.robot_radius, self.width - 1 - self.robot_radius),
                          y=randint(self.robot_radius, self.height - 1 - self.robot_radius),
                          speed=self.robot_speed,
                          radius=self.robot_radius,
                          noise_mu=self.noise_mu,
                          noise_musd=self.noise_musd,
                          noise_sd=self.noise_sd,
                          initial_reward=self.initial_reward,
                          fuel_cost=self.fuel_cost,
                          info_cost=self.info_cost,
                          behavior=SaboteurBehavior(),  # Line that changes
                          environment=self,
                          communication_cooldown=self.robot_comm_cooldown,
                          communication_stop_time=self.robot_comm_stop_time)
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
            if norm(bot.pos - np.array([x, y]).astype('float64')) < self.robot_radius:
                selected = bot
                break

        return selected

    def create_spawn_dicts(self):
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
