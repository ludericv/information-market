import os

from src.main_controller import MainController
from multiprocessing import Process, cpu_count


def main():
    controller = MainController(config_file="../config/config.txt")


def main_processes():
    NB_RUNS = 128
    N_CORES = cpu_count()

    process_table = [Process(target=run) for i in range(NB_RUNS)]
    for batch in range(NB_RUNS // N_CORES):
        for batch_process in range(N_CORES):
            process_table[batch_process + batch * N_CORES].start()
            print(f"launched process {batch_process + batch * N_CORES}")
        for batch_process in range(N_CORES):
            process_table[batch_process + batch * N_CORES].join()
            print(f"joined process {batch_process + batch * N_CORES}")
    for batch_process in range(NB_RUNS % N_CORES):
        process_table[batch_process + NB_RUNS - NB_RUNS % N_CORES].start()
        print(f"launched process {batch_process + NB_RUNS - NB_RUNS % N_CORES}")
    for batch_process in range(NB_RUNS % N_CORES):
        process_table[batch_process + NB_RUNS - NB_RUNS % N_CORES].join()
        print(f"joined process {batch_process + NB_RUNS - NB_RUNS % N_CORES}")


def run():
    controller = MainController(config_file="../config/config.txt")
    filename = "24careful_s3_1greedy.txt"
    with open(f"../data/behaviors/rewards/{filename}", "a") as file:
        file.write(controller.get_reward_stats())
    with open(f"../data/behaviors/items_collected/{filename}", "a") as file:
        file.write(controller.get_items_collected_stats())


if __name__ == '__main__':
    # main()
    main_processes()
