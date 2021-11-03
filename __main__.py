from main_controller import MainController
from multiprocessing import Process


def main():
    controller = MainController(config_file="config.txt")


def main_processes():
    NB_RUNS = 60

    process_table = [Process(target=run) for i in range(NB_RUNS)]
    for batch in range(NB_RUNS//4):
        for batch_process in range(4):
            process_table[batch_process + batch*4].start()
            print(f"launched process {batch_process + batch*4}")
        for batch_process in range(4):
            process_table[batch_process + batch*4].join()
            print(f"joined process {batch_process + batch*4}")
        print(f"end of batch {batch}")


def run():
    with open("results_weighteddecaying_expnoise.txt", "a") as file:
        controller = MainController(config_file="config.txt")
        file.write(controller.get_reward_stats())


if __name__ == '__main__':
    # main_processes()
    main()
