import tkinter as tk
import time


class ViewController:

    def __init__(self, controller, width=500, height=500, fps_cap=60):
        self.controller = controller
        self.fps_cap = fps_cap
        self.fps = fps_cap
        self.fps_update_counter = 0
        self.root = tk.Tk()
        self.root.title("Simulator")
        self.canvas = tk.Canvas(self.root, width=width, height=height, highlightthickness=0)
        self.canvas.configure(bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.last_frame_time = time.time()
        self.last_fps_check_time = time.time()

        self.update()
        self.root.mainloop()

    def update(self):
        # Update environment and display it on canvas
        self.canvas.delete("all")
        self.controller.step()
        self.controller.environment.draw(self.canvas)

        # Count elapsed time and schedule next update
        diff = time.time() - self.last_frame_time
        self.last_frame_time = time.time()
        self.canvas.create_text(50, 10, fill="black",
                                text=f"{round(self.fps)} FPS - step {self.controller.tick}")
        remaining_duration = 1.0 / self.fps_cap - diff
        if remaining_duration < 0:
            remaining_duration = 0
        self.root.after(round(1000 * remaining_duration), self.update)

        # Update FPS counter
        self.fps_update_counter += 1
        if self.fps_update_counter % self.fps_cap == 0:
            self.fps_update_counter = 0
            duration = time.time()-self.last_fps_check_time
            self.fps = self.fps_cap/duration
            self.last_fps_check_time = time.time()




