import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    dec_const = pd.read_csv("../data/old_data/results_decaying_constant.txt", header=None).values.flatten()
    w_dec_const = pd.read_csv("../data/old_data/results_weighteddecaying_constant.txt", header=None).values.flatten()
    better = pd.read_csv("../data/old_data/results_betterage.txt", header=None).values.flatten()
    w_better = pd.read_csv("../data/old_data/results_weightedbetterage.txt", header=None).values.flatten()
    dec_lin = pd.read_csv("../data/old_data/results_decaying_linearnoise.txt", header=None).values.flatten()
    w_dec_lin = pd.read_csv("../data/old_data/results_weighteddecaying_linearnoise.txt", header=None).values.flatten()
    dec_exp = pd.read_csv("../data/old_data/results_decaying_expnoise.txt", header=None).values.flatten()
    w_dec_exp = pd.read_csv("../data/old_data/results_weighteddecaying_expnoise.txt", header=None).values.flatten()
    dec_exp_const = pd.read_csv("../data/old_data/results_decaying_expconst.txt", header=None).values.flatten()
    # print(dec_const.mean(axis=1).mean())
    # print(better.mean(axis=1).mean())
    # print(dec_lin.mean(axis=1).mean())
    fig, axs = plt.subplots(1, 6, sharey=True)
    fig.set_size_inches(12, 6)
    pd.DataFrame(dec_const).boxplot(ax=axs[0]).set_title("Decaying Constant")
    pd.DataFrame(w_dec_const).boxplot(ax=axs[1]).set_title("W-Decaying Constant")
    # pd.DataFrame(better).boxplot(ax=axs[2]).set_title("Better Age")
    # pd.DataFrame(w_better).boxplot(ax=axs[3]).set_title("W-Better Age")
    pd.DataFrame(dec_lin).boxplot(ax=axs[2]).set_title("Decaying Linear")
    pd.DataFrame(w_dec_lin).boxplot(ax=axs[3]).set_title("W-Decaying Linear")
    pd.DataFrame(dec_exp).boxplot(ax=axs[4]).set_title("Decaying Exp")
    pd.DataFrame(w_dec_exp).boxplot(ax=axs[5]).set_title("W-Decaying Exp")
    # pd.DataFrame(dec_exp_const).boxplot(ax=axs[5]).set_title("Dec Exp Const")
    plt.show()


def main2():
    df = pd.read_csv("../data/behaviors/rewards/25honest.txt", header=None)
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
                df = pd.read_csv(f"../data/quality_comp/{f1}/{f2}/{s}.txt").values.flatten()
                pd.DataFrame(df).boxplot(ax=axs[i]).set_title(s)
            plt.show()


def compare_behaviors():
    filenames = ["25careful", "25smart_t25",  # "25smart_t25",
                 "24careful_s3_1saboteur", "24smart_t25_1saboteur",  # "24smart_t25_1greedy",
                 "23careful_2saboteur", "23smart_t25_2saboteur",  # "23smart_t25_2greedy",
                 "22careful_3saboteur", "22smart_t25_3saboteur",  # "22smart_t25_3greedy",
                 "20careful_s3_5saboteur", "20smart_t25_5saboteur"  # , "20smart_t25_5greedy"
                 ]
    show_run_difference(filenames, by=2)
    filenames = ["25honest", "25careful", "25smart_t25",
                 "24honest_1saboteur", "24careful_s3_1saboteur", "24smart_t25_1saboteur"]
    rewards_plot(filenames)
    show_run_difference(filenames, by=3)
    filenames = [
        "25honest", "25smart_t10", "25smart_t25", "25smart_t40_0saboteur", "25smart_t100_0saboteur",
        "24honest_1saboteur", "24smart_t10_1saboteur", "24smart_t25_1saboteur", "24smart_t40_1saboteur",
        "24smart_t100_1saboteur"
    ]
    show_run_difference(filenames, by=5)
    filenames = ["24smart_t25_1greedy", "24smart_t25_1greedy_minus10",
                 "22smart_t25_3greedy", "22smart_t25_3greedy_minus10",
                 "20smart_t25_5greedy", "20smart_t25_5greedy_minus10"]
    show_run_difference(filenames, by=2)


def honest_vs_careful():
    h_name = "24honest_1saboteur"
    c_name = "24careful_s3_1saboteur"
    hdf = pd.read_csv(f"../data/behaviors/rewards/{h_name}.txt", header=None)
    cdf = pd.read_csv(f"../data/behaviors/rewards/{c_name}.txt", header=None)
    hdiff = hdf.iloc[:, :24].apply(np.mean, axis=1) - hdf.iloc[:, 24]
    cdiff = cdf.iloc[:, :24].apply(np.mean, axis=1) - cdf.iloc[:, 24]
    fig, axs = plt.subplots()
    axs.set_title("Reward Difference with Saboteur")
    line_hist(hdiff.values.flatten(), 1.5)
    line_hist(cdiff.values.flatten(), 1.5)
    axs.legend(["honest", "careful"])
    plt.show()


def items_collected_plot(filenames):
    fig, axs = plt.subplots()
    for filename in filenames:
        collected = pd.read_csv(f"../data/behaviors/items_collected/{filename}.txt", header=None)
        n_honest = int(re.search('[0-9]+', filename).group())
        line_hist(collected.iloc[:, :n_honest].values.flatten(), 1)
    axs.legend(filenames)
    axs.set_title("Items Collected Distribution")
    plt.show()


def rewards_plot(filenames):
    fig, axs = plt.subplots()
    for filename in filenames:
        rewards = pd.read_csv(f"../data/behaviors/rewards/{filename}.txt", header=None)
        n_honest = int(re.search('[0-9]+', filename).group())
        line_hist(rewards.iloc[:, :n_honest].values.flatten(), 1)
    axs.legend(filenames)
    axs.set_title("Honest Rewards Distribution")
    plt.show()


def show_run_difference(filenames, by=1, metric="items_collected"):
    nrows = len(filenames) // by
    ncols = by
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True)
    print(axs)
    fig.set_size_inches(8 * ncols, 6 * nrows)
    for i, filename in enumerate(filenames):
        row = i // by
        col = i % by
        df = pd.read_csv(f"../data/behaviors/{metric}/{filename}.txt", header=None)
        n_honest = int(re.search('[0-9]+', filename).group())
        n_bad = 25 - n_honest
        axs[row, col].set_title(filename)
        axs[row, col].set_ylabel("type")
        axs[row, col].set_xlabel("run")
        means = df.iloc[:, :n_honest].apply(np.mean, axis=1)
        means_bad = df.iloc[:, -n_bad:].apply(np.mean, axis=1)
        stds = df.iloc[:, :n_honest].apply(np.std, axis=1)
        stds_bad = df.iloc[:, -n_bad:].apply(np.std, axis=1)
        res = pd.concat([means, stds, means_bad, stds_bad], axis=1, keys=['mean', 'sd', 'mean_b', 'sd_b'])
        res = res.sort_values(by='mean')
        axs[row, col].errorbar(range(res.shape[0]), res['mean'], res['sd'], linestyle='None', marker='^')
        if n_bad > 0:
            axs[row, col].errorbar(range(res.shape[0]), res['mean_b'], res['sd_b'], linestyle='None', marker='^',
                                   c=(1, 0, 0, 1))
    plt.show()


def rewards_per_run(filename):
    df = pd.read_csv(f"../data/behaviors/rewards/{filename}.txt", header=None)
    n_honest = int(re.search('[0-9]+', filename).group())
    fig, axs = plt.subplots()
    for row in range(df.shape[0]):
        values = df.iloc[row, :n_honest].values.flatten()
        bins = np.arange(np.floor(np.min(values)), np.ceil(np.max(values)), 1)
        # plt.hist(values, bins=bins, fc=(1, 0, 0, 0.1))
        line_hist(values, 1, 0.1, color=(0.5, 0, 0, 0.1))
    plt.show()


def line_hist(values, precision, alpha=1.0, color=None):
    bins = np.arange(np.floor(np.min(values)), np.ceil(np.max(values)), precision)
    n, bins = np.histogram(values, bins=bins, density=True)
    plt.plot(bins[:-1], n, alpha=alpha, c=color)


if __name__ == '__main__':
    compare_behaviors()
