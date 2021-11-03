import pandas as pd
import matplotlib.pyplot as plt


def main():
    dec_const = pd.read_csv("results_betterage60.txt", header=None).values.flatten()
    w_dec_const = pd.read_csv("results_w-betterage60.txt", header=None).values.flatten()
    #better = pd.read_csv("results_betterage.txt", header=None).values.flatten()
    #w_better = pd.read_csv("results_weightedbetterage.txt", header=None).values.flatten()
    dec_lin = pd.read_csv("results_lindecay60.txt", header=None).values.flatten()
    w_dec_lin = pd.read_csv("results_w-lindecay60.txt", header=None).values.flatten()
    dec_exp = pd.read_csv("results_expdecay60.txt", header=None).values.flatten()
    w_dec_exp = pd.read_csv("results_w-expdecay60.txt", header=None).values.flatten()
    #dec_exp_const = pd.read_csv("results_decaying_expconst.txt", header=None).values.flatten()
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
    #pd.DataFrame(dec_exp_const).boxplot(ax=axs[5]).set_title("Dec Exp Const")
    plt.show()


if __name__ == '__main__':
    main()
