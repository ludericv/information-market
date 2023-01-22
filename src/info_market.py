import time

from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Process, cpu_count, Pool
import pathlib
from sys import argv


def main():
    config = Configuration(config_file=argv[1])
    if config.value_of("visualization")['activate']:
        main_controller = MainController(config)
        view_controller = ViewController(main_controller,
                                         config.value_of("width"),
                                         config.value_of("height"),
                                         config.value_of("visualization")['fps'])
    else:
        for arg in argv[1:]:
            config = Configuration(config_file=arg)
            run_processes(config)


def run_processes(config: Configuration):
    nb_runs = config.value_of("number_runs")
    start = time.time()
    with Pool() as pool:
        pool.starmap(run, [(config, i) for i in range(nb_runs)])
    print(f'Finished {nb_runs} runs in {time.time()-start: .02f} seconds.')


def run(config, i):
    print(f"launched process {i+1}")
    controller = MainController(config)
    nb_honest = config.value_of("behaviors")[0]['population_size']
    nb_saboteur = config.value_of("behaviors")[1]['population_size']
    lie_angle = config.value_of("behaviors")[1]['parameters']['rotation_angle']
    controller.start_simulation()
    filename = f"{nb_honest}sceptical_t25_{nb_saboteur}scaboteur_nopenalisation_rotation_{lie_angle}"
    with open(f"../data/scaboteur_rotation/rewards/{filename}.txt", "a") as file:
        file.write(controller.get_reward_stats())
    with open(f"../data/scaboteur_rotation/items_collected/{filename}.txt", "a") as file:
        file.write(controller.get_items_collected_stats())
    # with open(f"../data/scaboteur_rotation/drifts/{filename}.txt", "a") as file:
    #     file.write(controller.get_drift_stats())
    if config.value_of("data_collection")["precision_recording"]:
        path = f"../data/scaboteur_rotation/reward_evolution/{filename}"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/{i}.txt", "a") as file:
            file.write(controller.get_rewards_evolution())


if __name__ == '__main__':
    main()
