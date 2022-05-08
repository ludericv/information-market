import copy

from helpers import random_walk as rw
from random import random, choices, gauss
from math import sin, cos, radians
from tkinter import LAST
from collections import deque

from model.behavior import State
from model.communication import CommunicationSession
from model.navigation import Location
import numpy as np

from helpers.utils import get_orientation_from_vector, rotate, InsufficientFundsException, CommunicationState


class AgentAPI:
    def __init__(self, agent):
        self.speed = agent.speed
        self.carries_food = agent.carries_food
        self.radius = agent.radius
        self.get_vector = agent.get_vector
        self.get_levi_turn_angle = agent.get_levi_turn_angle
        self.get_mu = agent.noise_mu


class Agent:
    colors = {State.EXPLORING: "gray35", State.SEEKING_FOOD: "orange", State.SEEKING_NEST: "green"}

    def __init__(self, robot_id, x, y, speed, radius,
                 noise_mu, noise_musd, noise_sd, fuel_cost,
                 initial_reward, info_cost, behavior, environment, communication_cooldown,
                 communication_stop_time):

        self.id = robot_id
        self.pos = np.array([x, y]).astype('float64')

        self._speed = speed
        self._radius = radius
        self.items_collected = 0
        self._carries_food = False

        self._communication_cooldown = communication_cooldown
        self._comm_stop_time = communication_stop_time
        self._time_since_last_comm = self._comm_stop_time + self._communication_cooldown + 1
        self.comm_state = CommunicationState.OPEN

        self.orientation = random() * 360  # 360 degree angle
        self.noise_mu = gauss(noise_mu, noise_musd)
        if random() >= 0.5:
            self.noise_mu = -self.noise_mu
        self.noise_sd = noise_sd

        self.fuel_cost = fuel_cost
        self.info_cost = info_cost
        self.environment = environment

        self.levi_counter = 1
        self.trace = deque(self.pos, maxlen=100)

        self.behavior = behavior
        self.api = AgentAPI(self)

    def __str__(self):
        return f"ID: {self.id}\n" \
               f"state: {self.comm_state.name}\n" \
               f"expected food at: ({round(self.pos[0] + rotate(self.behavior.navigation_table.get_location_vector(Location.FOOD), self.orientation)[0])}, {round(self.pos[1] + rotate(self.behavior.navigation_table.get_location_vector(Location.FOOD), self.orientation)[1])}), \n" \
               f"   known: {self.behavior.navigation_table.is_location_known(Location.FOOD)}\n" \
               f"expected nest at: ({round(self.pos[0] + rotate(self.behavior.navigation_table.get_location_vector(Location.NEST), self.orientation)[0])}, {round(self.pos[1] + rotate(self.behavior.navigation_table.get_location_vector(Location.NEST), self.orientation)[1])}), \n" \
               f"   known: {self.behavior.navigation_table.is_location_known(Location.NEST)}\n" \
               f"info age:\n" \
               f"   -food={round(self.behavior.navigation_table.get_target(Location.FOOD).age, 3)}\n" \
               f"   -nest={round(self.behavior.navigation_table.get_target(Location.NEST).age, 3)}\n" \
               f"carries food: {self._carries_food}\n" \
               f"drift: {round(self.noise_mu, 5)}\n" \
               f"reward: {round(self.environment.payment_database.get_reward(self.id), 3)}$\n" \
               f"item count: {self.items_collected}\n" \
               f"dr: {np.round(self.behavior.get_dr(), 2)}\n" \
               f"{self.behavior.debug_text()}"

    def __repr__(self):
        return f"bot {self.id}"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not (self == other)

    def communicate(self, neighbors):
        self.previous_nav = copy.deepcopy(self.behavior.navigation_table)
        if self.comm_state == CommunicationState.OPEN:
            session = CommunicationSession(self, neighbors, self.info_cost)
            self.behavior.communicate(session)
        self.new_nav = self.behavior.navigation_table
        self.behavior.navigation_table = self.previous_nav

    def step(self):
        self.behavior.navigation_table = self.new_nav
        sensors = self.environment.get_sensors(self)
        # self.check_food(sensors)
        if not self.comm_state == CommunicationState.PROCESSING:
            self.behavior.step(sensors, AgentAPI(self))
            try:
                self.environment.payment_database.apply_cost(self.id, self.fuel_cost)
                self.move()
            except InsufficientFundsException:
                pass
        self.update_communication_state()
        self.update_trace()

    def get_target_from_behavior(self, location):
        return self.behavior.get_target(location)

    def get_target(self, location):
        return self.behavior.navigation_table.get_target(location)

    def get_nav_location_age(self, location):
        return self.behavior.navigation_table.get_age(location)

    def get_location_vector(self, location):
        return self.behavior.navigation_table.get_location_vector(location)

    def knows_location(self, location):
        return self.behavior.navigation_table.is_location_known(location)

    def update_trace(self):
        self.trace.appendleft(self.pos[1])
        self.trace.appendleft(self.pos[0])

    def is_stationary(self):
        trace_length = len(self.trace) // 2
        tolerance = 3
        if trace_length < 4 * tolerance:  # If robot hasn't done 2*tolerance steps yet, consider it is moving
            return False
        trace_x = [self.trace[2 * i] for i in range(min(trace_length, 4 * tolerance))]
        trace_y = [self.trace[2 * i + 1] for i in range(min(trace_length, 4 * tolerance))]
        max_x_gap = max(trace_x) - min(trace_x)
        max_y_gap = max(trace_y) - min(trace_y)

        return max_x_gap < tolerance * self._speed and max_y_gap < tolerance * self._speed

    def get_vector(self, location: Location):
        if self.environment.get_sensors(self)[location]:
            return rotate(self.environment.get_location(location, self) - self.pos, -self.orientation)
        else:
            raise Exception(f"Robot does not sense {location}")

    def move(self):
        wanted_movement = rotate(self.behavior.get_dr(), self.orientation)
        noise_angle = gauss(self.noise_mu, self.noise_sd)
        noisy_movement = rotate(wanted_movement, noise_angle)
        self.orientation = get_orientation_from_vector(noisy_movement)
        self.pos = self.clamp_to_map(self.pos + noisy_movement)

    def clamp_to_map(self, new_position):
        if new_position[0] < self._radius:
            new_position[0] = self._radius
        if new_position[1] < self._radius:
            new_position[1] = self._radius
        if new_position[0] > self.environment.width - self._radius:
            new_position[0] = self.environment.width - self._radius
        if new_position[1] > self.environment.height - self._radius:
            new_position[1] = self.environment.height - self._radius
        return new_position

    def update_levi_counter(self):
        self.levi_counter -= 1
        if self.levi_counter <= 0:
            self.levi_counter = choices(range(1, rw.get_max_levi_steps() + 1), rw.get_levi_weights())[0]

    def get_levi_turn_angle(self):
        angle = 0
        if self.levi_counter <= 1:
            angle = choices(np.arange(0, 360), rw.get_crw_weights())[0]
        self.update_levi_counter()
        return angle

    def draw(self, canvas):
        circle = canvas.create_oval(self.pos[0] - self._radius,
                                    self.pos[1] - self._radius,
                                    self.pos[0] + self._radius,
                                    self.pos[1] + self._radius,
                                    fill=self.behavior.color,
                                    outline=self.colors[self.behavior.state],
                                    width=3)
        self.draw_comm_radius(canvas)
        # self.draw_goal_vector(canvas)
        self.draw_orientation(canvas)
        # self.draw_trace(canvas)

    def draw_trace(self, canvas):
        tail = canvas.create_line(*self.trace)

    def draw_comm_radius(self, canvas):
        circle = canvas.create_oval(self.pos[0] - self.environment.robot_communication_radius,
                                    self.pos[1] - self.environment.robot_communication_radius,
                                    self.pos[0] + self.environment.robot_communication_radius,
                                    self.pos[1] + self.environment.robot_communication_radius,
                                    outline="gray")

    def draw_goal_vector(self, canvas):
        arrow = canvas.create_line(self.pos[0],
                                   self.pos[1],
                                   self.pos[0] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.FOOD),
                                       self.orientation)[0],
                                   self.pos[1] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.FOOD),
                                       self.orientation)[1],
                                   arrow=LAST,
                                   fill="black")
        arrow = canvas.create_line(self.pos[0],
                                   self.pos[1],
                                   self.pos[0] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.NEST),
                                       self.orientation)[0],
                                   self.pos[1] + rotate(
                                       self.behavior.navigation_table.get_location_vector(Location.NEST),
                                       self.orientation)[1],
                                   arrow=LAST,
                                   fill="black")

    def draw_orientation(self, canvas):
        line = canvas.create_line(self.pos[0],
                                  self.pos[1],
                                  self.pos[0] + self._radius * cos(radians(self.orientation)),
                                  self.pos[1] + self._radius * sin(radians(self.orientation)),
                                  fill="white")

    def speed(self):
        return self._speed

    def radius(self):
        return self._radius

    def reward(self):
        return self.environment.payment_database.get_reward(self.id)

    def carries_food(self):
        return self._carries_food

    def drop_food(self):
        self._carries_food = False
        self.items_collected += 1

    def pickup_food(self):
        self._carries_food = True

    def record_transaction(self, transaction):
        transaction.timestep = self.environment.timestep
        self.environment.payment_database.record_transaction(transaction)

    def update_communication_state(self):
        self._time_since_last_comm += 1
        if self._time_since_last_comm > self._comm_stop_time + self._communication_cooldown:
            self.comm_state = CommunicationState.OPEN
        elif self._time_since_last_comm > self._comm_stop_time:
            self.comm_state = CommunicationState.ON_COOLDOWN
        else:
            self.comm_state = CommunicationState.PROCESSING

    def communication_happened(self):
        self._time_since_last_comm = 0
