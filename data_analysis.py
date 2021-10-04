import pandas as pd


def main():
    ba = pd.read_csv("results_betterage.txt", header=None)
    wa = pd.read_csv("results_weightedaverage.txt", header=None)
    q = pd.read_csv("results_quality.txt", header=None)
    dq = pd.read_csv("results_decayingquality.txt", header=None)
    print(ba.mean(axis=1).mean())
    print(wa.mean(axis=1).mean())
    print(q.mean(axis=1).mean())
    print(dq.mean(axis=1).mean())


if __name__ == '__main__':
    main()
