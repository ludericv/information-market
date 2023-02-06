import time

import pandas as pd

from controllers.main_controller import MainController, Configuration
from controllers.view_controller import ViewController
from multiprocessing import Pool
from pathlib import Path
from os.path import join
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
        controllers = pool.starmap(run, [(config, i) for i in range(nb_runs)])
        record_data(config, controllers)

    print(f'Finished {nb_runs} runs in {time.time()-start: .02f} seconds.')


def record_data(config:Configuration, controllers):
    output_directory = config.value_of("data_collection")["output_directory"]
    filename = config.value_of("data_collection")["filename"]
    '''
    TODO suggested modification: use the config file to define the filename
    '''
    if filename is None or filename == "":
        n_naive=config.value_of("behaviors")[0]['population_size']
        n_sceptical=config.value_of("behaviors")[1]['population_size']
        n_honest=n_naive+n_sceptical
        n_saboteur=config.value_of("behaviors")[2]['population_size']
        n_scaboteur=config.value_of("behaviors")[3]['population_size']
        n_dishonest=n_saboteur+n_scaboteur
        #arbitrary threshold for the naives
        th_honest= 3000 if n_sceptical==0 and n_honest>0 \
            else config.value_of("behaviors")[1]['parameters']['threshold']
        text_th_honest=str(th_honest).replace(".","").replace(",","")#eg 0.5 -> 05
        lie_angle = config.value_of("behaviors")[3]['parameters']['rotation_angle'] if n_scaboteur >0 \
            else config.value_of("behaviors")[2]['parameters']['rotation_angle']
        lie_angle  
        penalization="no" if config.value_of("payment_system")["class"]=="DelayedPaymentPaymentSystem" else ""
        #TODO rework name convention to include more characteristics
        filename = \
            f"{n_honest}sceptical_{text_th_honest}th_"+\
            f"{n_dishonest}scaboteur_{lie_angle}rotation_"+\
            f"{penalization}penalisation.csv"
    
    for metric in config.value_of("data_collection")["metrics"]:
        match metric:
            case "rewards":
                rewards_df = pd.DataFrame([controller.get_rewards() for controller in controllers])
                Path(join(output_directory, "rewards")).mkdir(parents=True, exist_ok=True)
                rewards_df.to_csv(join(output_directory, "rewards", filename), index=False, header=False)
            case "items_collected":
                items_collected_df = pd.DataFrame([controller.get_items_collected() for controller in controllers])
                Path(join(output_directory, "items_collected")).mkdir(parents=True, exist_ok=True)
                items_collected_df.to_csv(join(output_directory, "items_collected", filename), index=False, header=False)
            case "drifts":
                drifts_df = pd.DataFrame([controller.get_drifts() for controller in controllers])
                Path(join(output_directory, "drifts")).mkdir(parents=True, exist_ok=True)
                drifts_df.to_csv(join(output_directory, "drifts", filename), index=False, header=False)
            case "rewards_evolution":
                dataframes = []
                for i, controller in enumerate(controllers):
                    df = pd.DataFrame(controller.get_rewards_evolution_list(), columns=["tick", "rewards_list"])
                    df["simulation_id"] = i
                    df = df.set_index("simulation_id")
                    dataframes.append(df)
                Path(join(output_directory, "rewards_evolution")).mkdir(parents=True, exist_ok=True)
                pd.concat(dataframes).to_csv(join(output_directory, "rewards_evolution", filename))
            case _:
                print(f"[WARNING] Could not record metric: '{metric}'. Metric name is not valid.")


def run(config, i):
    print(f"launched process {i+1}")
    controller = MainController(config)
    controller.start_simulation()
    return controller


if __name__ == '__main__':
    main()
