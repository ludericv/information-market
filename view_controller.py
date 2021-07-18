import tkinter as tk
import time


class ViewController:

    def __init__(self, width=500, height=500, fps=60):
        self.fps = fps
        self.root = tk.Tk()
        self.root.title("Simulator")
        self.canvas = tk.Canvas(self.root, width=width, height=height, highlightthickness=0)
        self.canvas.configure(bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.last_frame_time = time.time()

    def update(self, environment, step_nb):
        self.canvas.delete("all")
        environment.draw(self.canvas)
        self.root.update()
        now = time.time()
        diff = now - self.last_frame_time
        self.canvas.create_text(50, 10, fill="black",
                                text=f"{round(1 / diff) if diff > 1 / self.fps else self.fps} FPS - step {step_nb}")
        self.root.update()
        if diff < 1/self.fps:
            time.sleep((1/self.fps)-diff)

        self.last_frame_time = time.time()

