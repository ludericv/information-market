from main_controller import MainController
from multiprocessing import Process


def main():
    controller = MainController(config_file="config.txt")


def main_processes():
    NB_RUNS = 20

    thread_table = [Process(target=run) for i in range(NB_RUNS)]
    for t in thread_table:
        t.start()
    for t in thread_table:
        t.join()


def run():
    with open("results_decayingquality.txt", "a") as file:
        controller = MainController(config_file="config.txt")
        file.write(controller.get_reward_stats())


if __name__ == '__main__':
    main()
