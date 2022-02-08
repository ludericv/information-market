from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Process, cpu_count


def main():
    with open(file="../config/config.txt") as file:
        pass
    config = Configuration(config_file="../config/config.txt")
    if config.parameters["VISUALIZE"] != 0:
        main_controller = MainController(config)
        view_controller = ViewController(main_controller,
                                         config.parameters["WIDTH"],
                                         config.parameters["HEIGHT"],
                                         config.parameters["FPS"])
    else:
        run_processes(config)


def run_processes(config: Configuration):
    NB_RUNS = config.parameters["NB_RUNS"]
    N_CORES = cpu_count()

    process_table = [Process(target=run, args=(config,)) for i in range(NB_RUNS)]
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


def run(config):
    controller = MainController(config)
    controller.start_simulation()
    filename = "20honest_s3_5saboteur.txt"
    with open(f"../data/behaviors/rewards/{filename}", "a") as file:
        file.write(controller.get_reward_stats())
    with open(f"../data/behaviors/items_collected/{filename}", "a") as file:
        file.write(controller.get_items_collected_stats())


if __name__ == '__main__':
    main()
