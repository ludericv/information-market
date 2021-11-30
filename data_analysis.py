import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    better = pd.read_csv("results_betterage60.txt", header=None).values.flatten()
    w_better = pd.read_csv("results_w-betterage60.txt", header=None).values.flatten()
    dec_lin = pd.read_csv("results_lindecay60.txt", header=None).values.flatten()
    w_dec_lin = pd.read_csv("results_w-lindecay60.txt", header=None).values.flatten()
    dec_exp = pd.read_csv("results_expdecay60.txt", header=None).values.flatten()
    w_dec_exp = pd.read_csv("results_w-expdecay60.txt", header=None).values.flatten()
    # print(dec_const.mean(axis=1).mean())
    # print(better.mean(axis=1).mean())
    # print(dec_lin.mean(axis=1).mean())
    fig, axs = plt.subplots(1, 6, sharey=True)
    fig.set_size_inches(12, 6)
    pd.DataFrame(better).boxplot(ax=axs[0]).set_title("Better Age")
    pd.DataFrame(w_better).boxplot(ax=axs[1]).set_title("W-Better Age")
    pd.DataFrame(dec_lin).boxplot(ax=axs[2]).set_title("Decaying Linear")
    pd.DataFrame(w_dec_lin).boxplot(ax=axs[3]).set_title("W-Decaying Linear")
    pd.DataFrame(dec_exp).boxplot(ax=axs[4]).set_title("Decaying Exp")
    pd.DataFrame(w_dec_exp).boxplot(ax=axs[5]).set_title("W-Decaying Exp")
    #pd.DataFrame(dec_exp_const).boxplot(ax=axs[5]).set_title("Dec Exp Const")
    plt.show()


def main2():
    df = pd.read_csv("data_sorted.txt", header=None)
    plt.plot(df.apply(np.mean, axis=0))
    plt.show()

def compare_strats():
    strats = ["age", "w-age", "lin", "w-lin", "exp", "w-exp"]
    mu_folders = ["low_0", "med_05", "high_10"]
    sd_folders = ["low_01", "high_10"]
    popsize_folders = {"low_10", "high_50"}
    d = {"mu": mu_folders, "sd": sd_folders, "popsize": popsize_folders}
    for f1 in d:
        for f2 in d[f1]:
            fig, axs = plt.subplots(1, len(strats), sharey=True)
            fig.set_size_inches(16, 6)
            fig.suptitle(f"{f1}: {f2}")
            for i, s in enumerate(strats):
                df = pd.read_csv(f"data/quality_comp/{f1}/{f2}/{s}.txt").values.flatten()
                pd.DataFrame(df).boxplot(ax=axs[i]).set_title(s)
            plt.show()


if __name__ == '__main__':
    # main()
    # main2()
    compare_strats()