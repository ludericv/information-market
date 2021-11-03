from main_controller import MainController
from multiprocessing import Process


def main():
    controller = MainController(config_file="config.txt")


def main_processes():
    NB_RUNS = 60
    N_CORES = 8

    process_table = [Process(target=run) for i in range(NB_RUNS)]
    for batch in range(NB_RUNS//N_CORES):
        for batch_process in range(N_CORES):
            process_table[batch_process + batch*N_CORES].start()
            print(f"launched process {batch_process + batch*N_CORES}")
        for batch_process in range(N_CORES):
            process_table[batch_process + batch*N_CORES].join()
            print(f"joined process {batch_process + batch*N_CORES}")
    for batch_process in range(NB_RUNS%N_CORES):
        process_table[batch_process + NB_RUNS - NB_RUNS%N_CORES].start()
        print(f"launched process {batch_process + NB_RUNS - NB_RUNS%N_CORES}")
    for batch_process in range(NB_RUNS%N_CORES):
        process_table[batch_process + NB_RUNS - NB_RUNS%N_CORES].join()
        print(f"joined process {batch_process + NB_RUNS - NB_RUNS%N_CORES}")



def run():
    with open("results_lindecay60.txt", "a") as file:
        controller = MainController(config_file="config.txt")
        file.write(controller.get_reward_stats())


if __name__ == '__main__':
    main_processes()
    # main()
