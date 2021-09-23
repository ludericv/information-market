import pandas as pd
def main():
    ba = pd.read_csv("results_betterage.txt", header=None)
    dq = pd.read_csv("results_decayingquality.txt", header=None)
    print(ba.mean(axis=1).mean())
    print(dq.mean(axis=1).mean())

if __name__ == '__main__':
    main()
