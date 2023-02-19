import os
import tkinter as tk
import time

from PIL import Image, ImageTk


class ViewController:

    def __init__(self, controller, width=500, height=500, fps_cap=60):
        self.controller = controller
        self.fps_cap = fps_cap
        self.fps = fps_cap
        self.fps_update_counter = 0

        self.root = tk.Tk()
        self.controller.environment.load_images()
        self.root.title("Foraging Simulator")

        self.canvas = tk.Canvas(self.root, width=width, height=height, highlightthickness=0)
        self.canvas.configure(bg="white")
        self.canvas.pack(fill="both", expand=False, side="left")

        self.debug_canvas = tk.Canvas(self.root, width=200, height=height/2, highlightthickness=0)
        self.debug_canvas.configure(bg="white smoke")
        self.debug_canvas.pack(fill="both", expand=False, side="right")
        self.selected_robot = None
        debug_title = self.debug_canvas.create_text(5, 5, fill="gray30", text=f"Robot", font="Arial 13 bold", anchor="nw")
        self.debug_text = self.debug_canvas.create_text(5, 25, fill="gray30", text=f"No robot selected", anchor="nw", font="Arial 10")

        self.animation_ended = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.paused = False
        self.can_render = False
        self.create_bindings()

        self.last_frame_time = time.time()
        self.last_fps_check_time = time.time()
        self.start_animation()

    def start_animation(self):
        while not self.animation_ended:
            # Draw environment
            if self.can_render:
                self.last_frame_time = time.time()
                self.fps_update_counter += 1
                # Update environment
                if not self.paused:
                    self.controller.step()

                self.refresh()
                self.can_render = False

            # Count elapsed time and schedule next update
            if time.time() - self.last_frame_time >= 1.0 / self.fps_cap:
                self.can_render = True

            # Update FPS counter
            if time.time() - self.last_fps_check_time >= 1:
                self.fps = self.fps_update_counter
                self.fps_update_counter = 0
                self.last_fps_check_time = time.time()

            self.root.update()

    def refresh(self):
        self.display_selected_info()
        # self.update_stats_canvas()
        self.canvas.delete("all")

        self.controller.environment.draw(self.canvas)
        self.canvas.create_text(10, 10, fill="black",
                                text=f"{round(self.fps)} FPS - step {self.controller.clock.tick}", anchor="nw")

        if self.selected_robot is not None:
            circle = self.canvas.create_oval(self.selected_robot.pos[0] - self.selected_robot._radius,
                                             self.selected_robot.pos[1] - self.selected_robot._radius,
                                             self.selected_robot.pos[0] + self.selected_robot._radius,
                                             self.selected_robot.pos[1] + self.selected_robot._radius,
                                             outline="red", width=3)

    def create_bindings(self):
        self.root.bind("<space>", self.switch_animating_state)
        self.root.bind("<Button-1>", self.select_robot)
        self.root.bind("<n>", lambda event: self.controller.step())

    def on_closing(self):
        self.animation_ended = True

    def switch_animating_state(self, event):
        self.paused = not self.paused

    def select_robot(self, event):
        self.selected_robot = self.controller.get_robot_at(event.x, event.y)

    def display_selected_info(self):
        self.debug_canvas.delete(self.debug_text)
        if self.selected_robot is not None:
            self.debug_text = self.debug_canvas.create_text(5, 25, fill="gray45", text=self.selected_robot, anchor="nw", font="Arial 10")
        else:
            self.debug_text = self.debug_canvas.create_text(5, 25, fill="gray45", text=f"No robot selected", anchor="nw", font="Arial 10")
