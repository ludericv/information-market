from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Process, cpu_count
from sys import argv


def main():
    config = Configuration(config_file=argv[1])
    if config.parameters["VISUALIZE"] != 0:
        main_controller = MainController(config)
        view_controller = ViewController(main_controller,
                                         config.parameters["WIDTH"],
                                         config.parameters["HEIGHT"],
                                         config.parameters["FPS"])
    else:
        for arg in argv[1:]:
            config = Configuration(config_file=arg)
            run_processes(config)


def run_processes(config: Configuration):
    nb_runs = config.parameters["NB_RUNS"]
    nb_cores = cpu_count()

    process_table = [Process(target=run, args=(config,)) for i in range(nb_runs)]
    for batch in range(nb_runs // nb_cores):
        for batch_process in range(nb_cores):
            process_table[batch_process + batch * nb_cores].start()
            print(f"launched process {batch_process + batch * nb_cores}")
        for batch_process in range(nb_cores):
            process_table[batch_process + batch * nb_cores].join()
            print(f"joined process {batch_process + batch * nb_cores}")
    for batch_process in range(nb_runs % nb_cores):
        process_table[batch_process + nb_runs - nb_runs % nb_cores].start()
        print(f"launched process {batch_process + nb_runs - nb_runs % nb_cores}")
    for batch_process in range(nb_runs % nb_cores):
        process_table[batch_process + nb_runs - nb_runs % nb_cores].join()
        print(f"joined process {batch_process + nb_runs - nb_runs % nb_cores}")


def run(config):
    controller = MainController(config)
    nb_honest = config.parameters["NB_HONEST"]
    nb_saboteur = config.parameters["NB_ROBOTS"] - nb_honest
    controller.start_simulation()
    filename = f"{nb_honest}smart_t25_{nb_saboteur}freerider_30+10_infocost25.txt"
    with open(f"../data/stop_time/rewards/{filename}", "a") as file:
        file.write(controller.get_reward_stats())
    with open(f"../data/stop_time/items_collected/{filename}", "a") as file:
        file.write(controller.get_items_collected_stats())


if __name__ == '__main__':
    main()
