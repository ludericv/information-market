from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Process, cpu_count
import pathlib
from sys import argv


def main():
    config = Configuration(config_file=argv[1])
    if config.value_of("VISUALIZE") != 0:
        main_controller = MainController(config)
        view_controller = ViewController(main_controller,
                                         config.value_of("WIDTH"),
                                         config.value_of("HEIGHT"),
                                         config.value_of("FPS"))
    else:
        for arg in argv[1:]:
            config = Configuration(config_file=arg)
            run_processes(config)


def run_processes(config: Configuration):
    nb_runs = config.value_of("NB_RUNS")
    nb_cores = cpu_count()

    process_table = [Process(target=run, args=(config,i,)) for i in range(nb_runs)]
    for batch in range(nb_runs // nb_cores):
        for batch_process in range(nb_cores):
            process_nb = batch_process + batch * nb_cores
            process_table[process_nb].start()
            print(f"launched process {process_nb}")
        for batch_process in range(nb_cores):
            process_nb = batch_process + batch * nb_cores
            process_table[process_nb].join()
            print(f"joined process {process_nb}")
    for batch_process in range(nb_runs % nb_cores):
        process_nb = batch_process + nb_runs - nb_runs % nb_cores
        process_table[process_nb].start()
        print(f"launched process {process_nb}")
    for batch_process in range(nb_runs % nb_cores):
        process_nb = batch_process + nb_runs - nb_runs % nb_cores
        process_table[process_nb].join()
        print(f"joined process {process_nb}")


def run(config, i):
    controller = MainController(config)
    nb_honest = config.value_of("NB_HONEST")
    nb_saboteur = config.value_of("NB_ROBOTS") - nb_honest
    controller.start_simulation()
    filename = f"{nb_honest}smart_t25_{nb_saboteur}smartboteur_nostake"
    with open(f"../data/correlation/rewards/{filename}.txt", "a") as file:
        file.write(controller.get_reward_stats())
    with open(f"../data/correlation/items_collected/{filename}.txt", "a") as file:
        file.write(controller.get_items_collected_stats())
    with open(f"../data/correlation/drifts/{filename}.txt", "a") as file:
        file.write(controller.get_drift_stats())
    if config.value_of("RECORD") == 1:
        path = f"../data/correlation/reward_evolution/{filename}"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/{i}.txt", "a") as file:
            file.write(controller.get_rewards_evolution())


if __name__ == '__main__':
    main()
