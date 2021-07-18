from main_controller import MainController


def main():
    controller = MainController(config_file="config.txt")
    controller.start_simulation()


if __name__ == '__main__':
    main()
