from main_controller import MainController
from multiprocessing import Process, cpu_count


def main():
    controller = MainController(config_file="config.txt")


def main_processes():
    NB_RUNS = 4
    N_CORES = cpu_count()

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
    with open("results_weighteddecaying_expnoise.txt", "a") as file:
        controller = MainController(config_file="config.txt")
        file.write(controller.get_sorted_reward_stats())


if __name__ == '__main__':
    # main_processes()
    main()
