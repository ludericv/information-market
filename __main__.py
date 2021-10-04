from main_controller import MainController
from multiprocessing import Process


def main():
    controller = MainController(config_file="config.txt")


def main_processes():
    NB_RUNS = 20

    process_table = [Process(target=run) for i in range(NB_RUNS)]
    for batch in range(NB_RUNS//4):
        for batch_process in range(batch):
            process_table[batch_process + batch*4].start()
        for batch_process in range(batch):
            process_table[batch_process + batch*4].join()



def run():
    with open("results_weightedaverage.txt", "a") as file:
        controller = MainController(config_file="config.txt")
        file.write(controller.get_reward_stats())


if __name__ == '__main__':
    # main_processes()
    main()
