from environment import Environment

def main():
    env = Environment(config_file="config.txt")
    env.start_simulation()

if __name__ == '__main__':
    main()