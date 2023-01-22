# Master's Thesis: An Information Market for Social Navigation in Robots
## Requirements
To be able to run information-market, you must have:
- A recent version of Linux, MacOSX or Windows
- **Python** 3.6 or newer
## Installation
First clone the git repository into a given folder on your computer:
```bash
git clone https://github.com/ludericv/information-market.git
```
Then, navigate into the newly created folder, create a new python virtual environment and install the required python packages:
```bash
cd information-market
python -m venv infomarket-env
source infomarket-env/bin/activate
pip install -r requirements.txt
```
## Run
First, edit the `config.json` file inside the config folder with the parameters you want to use. Then, to run information-market, simply open a terminal, cd to the src folder and run the program.
```bash
cd path/to/src
python info_market.py ../config/config.json
```

## Configuration

A simulation's parameters are defined in a json configuration file (such as `config/config.json`). The parameters are the following:

- width: simulation environment's width
- height: simulation environment's height
- food: food area's position and radius
- nest: nest area's position and radius
- simulation_steps: simulation duration (number or time steps)
- number_runs: number of parallel simulation runs (only applicable when visualization is turned off)
- visualization: parameters related to the visualization
  - activate: whether to activate the simulation GUI
  - fps: maximum framerate of the simulation
- random_walk: random walk parameters
  - random_walk_factor: controls how much successive random walk turns are correlated
  - levi_factor: controls the duration before successive random walk turns
- agent: robot parameters (common to all robots regardless of behavior)
  - radius: robot radius
  - speed: robot speed
  - communication_radius: communication range
  - communication_stop_time: duration a robot must stop moving when communicating
  - communication_cooldown: how long a robot must wait between communications
  - noise_sampling_mu: odometric noise sampling distribution mu
  - noise_sampling_sigma: odometric noise sampling distribution sigma
  - noise_sd: odometric noise standard deviation (in degrees), i.e. how different the odometric noise at different time steps
  - fuel_cost: cost of moving at each time step (deducted from robot's monetary balance)
- behaviors: list of behaviors used in the simulation
  - class: name of the behavior class (from the code, see Behaviors section)
  - population_size: number of robots with the behavior
  - parameters: behavior-specific keyword arguments
- payment_system: payment system parameters
  - class: name of the payment system class (from the code, see Payment Systems section)
  - initial_reward: amount of money robots start the simulation with
  - parameters: payment-system-specific keyword arguments
- market: reward mechanism parameters
  - class: name of the market class (from the code, use "FixedPriceMarket")
  - parameters: market-specific parameters
    - reward: reward for selling a strawberry at the nest
- data_collection: parameters for data collection
  - precision_recording: whether to enable precision recording to save robot rewards at multiple points during the simulation
  - precision_recording_interval: resolution (in number of time steps) for the precision recording