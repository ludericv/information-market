import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model.market import exponential_model, logistics_model


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
    show_run_proportions(filenames, by=2)
    filenames = ["25honest", "25careful", "25smart_t25",
                 "24honest_1saboteur", "24careful_s3_1saboteur", "24smart_t25_1saboteur"]
    rewards_plot(filenames)
    show_run_proportions(filenames, by=3)
    filenames = [
        "25honest", "25smart_t10", "25smart_t25", "25smart_t40_0saboteur", "25smart_t100_0saboteur",
        "24honest_1saboteur", "24smart_t10_1saboteur", "24smart_t25_1saboteur", "24smart_t40_1saboteur",
        "24smart_t100_1saboteur"
    ]
    show_run_proportions(filenames, by=5)
    filenames = ["24smart_t25_1greedy", "24smart_t25_1greedy_minus10",
                 "22smart_t25_3greedy", "22smart_t25_3greedy_minus10",
                 "20smart_t25_5greedy", "20smart_t25_5greedy_minus10"]
    show_run_proportions(filenames, by=2)
    filenames = ["25honest", "25careful", "25smart_t25"]
    show_run_difference(filenames, by=3, metric="items_collected")
    filenames = ["24honest_1saboteur", "24careful_s3_1saboteur", "24smart_t25_1saboteur"]
    show_run_difference(filenames, by=3, metric="items_collected")


def powerpoint_plots():
    # filenames = ["24smart_t25_1greedy",
    #              "22smart_t25_3greedy",
    #              "20smart_t25_5greedy"]
    # show_run_proportions(filenames, by=3)
    # make_violin_plots(filenames, by=3)
    # filenames = ["24smart_t25_1greedy_SoR_50",
    #              "22smart_t25_3greedy_SoR_50",
    #              "20smart_t25_5greedy_SoR_50"]
    # make_violin_plots(filenames, by=3, comparison_on="payment_types")
    # show_run_proportions(filenames, by=3, comparison_on="payment_types")
    # filenames = ["24smart_t25_1saboteur",
    #              "22smart_t25_3saboteur",
    #              "20smart_t25_5saboteur"]
    # make_violin_plots(filenames, by=3)
    # show_run_proportions(filenames, by=3)
    # filenames = ["24smart_t25_1saboteur_SoR_50",
    #              "22smart_t25_3saboteur_SoR_50",
    #              "20smart_t25_5saboteur_SoR_50"]
    # make_violin_plots(filenames, by=3, comparison_on="payment_types")
    # show_run_proportions(filenames, by=3, comparison_on="payment_types")
    # filenames = ["24smart_t25_1smartboteur_SoR_50",
    #              "22smart_t25_3smartboteur_SoR_50",
    #              "20smart_t25_5smartboteur_SoR_50"]
    # make_violin_plots(filenames, by=3, comparison_on="saboteur_comp")
    # show_run_proportions(filenames, by=3, comparison_on="saboteur_comp")
    filenames = ["24smart_t25_1smartboteur_SoR_50_lastKmarket",
                 "22smart_t25_3smartboteur_SoR_50_lastKmarket",
                 "20smart_t25_5smartboteur_SoR_50_lastKmarket"]
    make_violin_plots(filenames, by=3, comparison_on="saboteur_comp")
    show_run_proportions(filenames, by=3, comparison_on="saboteur_comp")


def compare_payment_types():
    filenames = []
    for i in [25, 24, 22, 20]:
        j = 25 - i
        filenames.append(f"{i}smart_t25_{j}greedy_50_infocost_fixed")
        filenames.append(f"{i}smart_t25_{j}greedy_50_infocost_timevarying")
    show_run_proportions(filenames, by=2, comparison_on="payment_types", metric="rewards")
    show_run_proportions([filenames[i] for i in [1, 3, 5, 7]], by=4, comparison_on="payment_types", metric="rewards")
    make_violin_plots([filenames[i] for i in [3, 5, 7]], by=3, comparison_on="payment_types", metric="rewards")


def compare_stop_time():
    filenames = []
    for i in [25, 23, 20]:
        j = 25 - i
        for cd in [10, 30, 60]:
            filenames.append(f"{i}smart_t25_{j}freerider_{cd}+10")
    show_run_proportions(filenames, by=3, comparison_on="stop_time", metric="rewards")


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


def show_run_difference(filenames, by=1, comparison_on="behaviors", metric="rewards"):
    nrows = len(filenames) // by
    ncols = by
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True)
    print(axs)
    fig.set_size_inches(4 * ncols, 6 * nrows)
    for i, filename in enumerate(filenames):
        row = i // by
        col = i % by
        df = pd.read_csv(f"../data/{comparison_on}/{metric}/{filename}.txt", header=None)
        n_honest = int(re.search('[0-9]+', filename).group())
        n_bad = 25 - n_honest

        if nrows == 1 or ncols == 1:
            frame = axs[row] if ncols == 1 else axs[col]
        else:
            frame = axs[row, col]

        frame.set_title(filename)
        frame.set_ylabel(metric)
        frame.set_xlabel("run")
        run_difference_plot(df, frame, n_bad, n_honest)
    plt.show()


def run_difference_plot(df, ax, n_bad, n_honest):
    means = df.iloc[:, :n_honest].apply(np.mean, axis=1)
    means_bad = df.iloc[:, -n_bad:].apply(np.mean, axis=1)
    stds = df.iloc[:, :n_honest].apply(np.std, axis=1)
    stds_bad = df.iloc[:, -n_bad:].apply(np.std, axis=1)
    res = pd.concat([means, stds, means_bad, stds_bad], axis=1, keys=['mean', 'sd', 'mean_b', 'sd_b'])
    res = res.sort_values(by='mean')
    ax.errorbar(range(res.shape[0]), res['mean'], res['sd'], linestyle='None', marker='^', c="tab:blue")
    if n_bad > 0:
        ax.errorbar(range(res.shape[0]), res['mean_b'], res['sd_b'], linestyle='None', marker='^',
                    c=(1, 0, 0, 1))


def show_run_proportions(filenames, by=1, comparison_on="behaviors", metric="rewards"):
    nrows = len(filenames) // by
    ncols = by
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True)
    print(axs)
    fig.set_size_inches(4 * ncols, 6 * nrows)
    for i, filename in enumerate(filenames):
        row = i // by
        col = i % by
        df = pd.read_csv(f"../data/{comparison_on}/{metric}/{filename}.txt", header=None)
        n_honest = int(re.search('[0-9]+', filename).group())
        n_bad = 25 - n_honest
        honest_name = re.search('[a-z]+', filename.split("_")[0]).group()
        bad_name = ""
        if n_bad > 0:
            try:
                bad_name = re.search('[a-z]+', filename.split("_")[2]).group()
            except IndexError:
                bad_name = re.search('[a-z]+', filename.split("_")[1]).group()
        colors = {"smart": "blue", "honest": "blue", "careful": "blue",
                  "saboteur": "red",
                  "smartboteur": "red",
                  "greedy": "green"}
        if nrows == 1 and ncols == 1:
            frame = axs
        elif nrows == 1 or ncols == 1:
            frame = axs[row] if ncols == 1 else axs[col]
        else:
            frame = axs[row, col]

        totals = df.apply(np.sum, axis=1)
        frame.set_title(f"{filename}, throughput = {round(np.mean(totals))}")
        frame.set_ylabel(f"proportion of {metric} (%)")
        frame.set_xlabel("run")

        means = df.iloc[:, :n_honest].apply(np.mean, axis=1) * 100 / totals
        means_bad = df.iloc[:, -n_bad:].apply(np.mean, axis=1) * 100 / totals
        stds = df.iloc[:, :n_honest].apply(np.std, axis=1) * 100 / totals
        stds_bad = df.iloc[:, -n_bad:].apply(np.std, axis=1) * 100 / totals
        res = pd.concat([means, stds, means_bad, stds_bad], axis=1, keys=['mean', 'sd', 'mean_b', 'sd_b'])
        res = res.sort_values(by='mean')
        frame.errorbar(range(res.shape[0]), res['mean'], res['sd'], linestyle='None', marker='^', c=colors[honest_name])
        if n_bad > 0:
            frame.errorbar(range(res.shape[0]), res['mean_b'], res['sd_b'], linestyle='None', marker='^',
                           c=colors[bad_name])
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


def supply_demand_simulation():
    n_bots = 25
    demand = 0.3
    max_price = 3
    creditor_share = 0.5
    static_rewards = []
    supplies = range(1, n_bots + 1)
    active_rewards = []
    active_only_rewards = []
    for supply in supplies:
        n_active = supply
        reward_per_active = exponential_model(demand=demand * n_bots, max_price=max_price, supply=supply)
        static_reward = n_active * (creditor_share / (n_bots - 1)) * reward_per_active
        static_rewards.append(static_reward)
        active_reward = (((n_active - 1) / (n_bots - 1) * creditor_share) + 1 - creditor_share) * reward_per_active
        active_rewards.append(active_reward)
        active_only_rewards.append((1 - creditor_share) * reward_per_active)
    plt.plot(supplies, static_rewards)
    plt.plot(supplies, active_only_rewards)
    plt.plot(supplies, active_rewards)
    plt.axhline(y=max(static_rewards), color='r', linestyle='-')
    plt.legend(
        ["information", "foraging", "information + foraging", f"information max ={round(max(static_rewards), 2)}"])
    plt.title(f"creditor share = {creditor_share}, demand = {demand}")
    plt.xlabel("supply")
    plt.ylabel("reward")
    plt.show()


def make_violin_plots(filenames, by=1, comparison_on="behaviors", metric="rewards"):
    nrows = len(filenames) // by
    ncols = by
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharey=True, sharex=True)
    fig.set_size_inches(3 * ncols, 6 * nrows)
    palette = {
        "smart": "cornflowerblue",
        "greedy": "limegreen",
        "saboteur": "firebrick",
        "smartboteur": "firebrick"
    }
#    sns.set_palette(palette)

    for row in range(nrows):
        row_df = pd.DataFrame(columns=["proportion(%)", "behavior", "file"])
        if nrows == 1:
            frame = axs
        else:
            frame = axs[row]
        frame.set_title(f"Individual reward share across subpopulations")
        for col in range(ncols):
            filename = filenames[row * ncols + col]
            df = pd.read_csv(f"../data/{comparison_on}/{metric}/{filename}.txt", header=None)
            n_honest = int(re.search('[0-9]+', filename).group())
            honest_name = re.search('[a-z]+', filename.split("_")[0]).group()
            bad_name = re.search('[a-z]+', filename.split("_")[2]).group()
            n_bad = 25 - n_honest

            totals = df.apply(np.sum, axis=1)
            # print(totals)
            df_rel = df.div(totals, axis=0) * 100
            # means = df.iloc[:, :n_honest].apply(np.mean, axis=1) * 100 / totals
            honest_flat = pd.DataFrame(df_rel.iloc[:, :n_honest].to_numpy().flatten())
            bad_flat = pd.DataFrame(df_rel.iloc[:, -n_bad:].to_numpy().flatten())

            # means_bad = df.iloc[:, -n_bad:].apply(np.mean, axis=1) * 100 / totals
            # Draw a nested violinplot and split the violins for easier comparison

            goods = pd.DataFrame(np.full(honest_flat.shape, honest_name))
            bads = pd.DataFrame(np.full(honest_flat.shape, bad_name))
            honest_flat = pd.concat([honest_flat, goods], axis=1)
            bad_flat = pd.concat([bad_flat, bads], axis=1)
            final_df = pd.concat([honest_flat, bad_flat])
            final_df.columns = ["proportion(%)", "behavior"]
            final_df["file"] = f"{n_honest} {honest_name} vs {n_bad} {bad_name}"
            row_df = pd.concat([row_df, final_df])
        row_df.index = [i for i in range(row_df.shape[0])]
        temp = pd.DataFrame(row_df.to_dict())
        sns.violinplot(data=temp, x="file", y="proportion(%)", hue="behavior",
                       split=True, inner="quart", linewidth=1, ax=frame, palette=palette)
        sns.despine(left=True)

    plt.show()


def items_collected_violin_plot(df, frame, n_good, n_bad, good_name="naive", bad_name="saboteur"):
    palette = {
        "naive": "cornflowerblue",
        "smart": "cornflowerblue",
        "greedy": "limegreen",
        "saboteur": "firebrick",
        "smartboteur": "firebrick"
    }

    honest_flat = pd.DataFrame(df.iloc[:, :n_good].to_numpy().flatten())
    bad_flat = pd.DataFrame(df.iloc[:, -n_bad:].to_numpy().flatten())

    goods = pd.DataFrame(np.full(honest_flat.shape, good_name))
    bads = pd.DataFrame(np.full(honest_flat.shape, bad_name))
    honest_flat = pd.concat([honest_flat, goods], axis=1)
    bad_flat = pd.concat([bad_flat, bads], axis=1)
    final_df = pd.concat([honest_flat, bad_flat])
    final_df.columns = ["items collected", "behavior"]
    print(final_df)
    sns.violinplot(data=final_df, y="items collected", hue="behavior",
                   split=True, inner="quart", linewidth=1, ax=frame)
    sns.despine(left=True)

def test_angles():
    df = pd.DataFrame([[1, 1, 0], [2, 4, 0], [1, 100, 0], [4, 359, 0]], columns=["seller", "angle", "alike"])
    angle_window = 5
    total_amount = 0.5
    df["alike"] = df.apply(func=lambda row: pd.DataFrame(
        [(df.iloc[:, 1] - row[1]) % 360,
         (row[1] - df.iloc[:, 1]) % 360]).
                           apply(min, axis=0).
                           sum(),
                           axis=1)
    # pd.DataFrame([(df.iloc[:, 1] - row[1]) % 360, (row[1] - df.iloc[:, 1]) % 360]).apply(min, axis=1).sum()
    # print(pd.DataFrame(
    #     [(df.iloc[:, 1] - df.iloc[1, 1]) % 360,
    #      (df.iloc[1, 1] - df.iloc[:, 1]) % 360]))
    # print(df)
    sellers_to_alike = df.groupby("seller").sum()
    print(sellers_to_alike)
    sellers_to_alike["alike"] = sellers_to_alike["alike"].sum() - sellers_to_alike["alike"]
    sellers_to_alike = sellers_to_alike.to_dict()["alike"]
    print(sellers_to_alike)
    # total_shares = sum(sellers_to_alike.values())
    # mapping = {seller: total_amount * sellers_to_alike[seller] / total_shares for seller in sellers_to_alike}
    # print(mapping)


def test_timedev():
    sorted_transactions = [[1, 1], [2, 4], [3, 4], [1, 7], [5, 12]]
    timestep = 20
    total_amount = 0.5
    sorted_transactions.append([-1, timestep])
    df = pd.DataFrame(sorted_transactions, columns=["seller", "time"])
    df["dt"] = df["time"].diff().shift(-1) + 1
    df = df[:-1]
    df["score"] = 1 / df["dt"]
    df["share"] = total_amount * df["score"] / df["score"].sum()
    mapping = df.groupby("seller").sum().to_dict()["share"]
    print(df)

    print(mapping)


def thesis_plots():
    # Chapter 8 - Section 1
    # filename = "../data/behaviors/items_collected/25honest.txt"
    # df = pd.read_csv(filename, header=None)
    # general_boxplot_data = df.values.flatten()
    # fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 4]})
    # fig.set_size_inches(10, 6)
    # sns.violinplot(x=["all" for e in general_boxplot_data], y=general_boxplot_data, ax=axs[0], bw=.3)
    # run_difference_plot(df, axs[1], 0, 25)
    # #run_difference_plot(pd.read_csv("../data/behaviors/items_collected/24honest_1saboteur.txt", header=None), axs[2], 1, 24)
    # fig.suptitle("Performance of Naïve Swarm")
    # fig.supylabel("Number of Items Collected")
    # fig.supxlabel("Run Considered")
    # plt.show()

    df = pd.read_csv("../data/behaviors/items_collected/24honest_1saboteur.txt", header=None)
    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 4]})
    fig.set_size_inches(10, 6)
    items_collected_violin_plot(df, axs[0], 24, 1)
    run_difference_plot(df, axs[1], 1, 24)
    plt.show()


if __name__ == '__main__':
    # supply_demand_simulation()
    # compare_behaviors()
    # compare_payment_types()
    # compare_stop_time()
    # show_run_proportions(["20smart_t25_5freerider_10+10"], comparison_on="stop_time")
    # make_violin_plots(["20smart_t25_5saboteur", "22smart_t25_3saboteur"], by=2)
    # powerpoint_plots()
    # test_angles()
    # test_timedev()
    thesis_plots()