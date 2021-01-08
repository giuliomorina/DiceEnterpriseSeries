import numpy as np
from DiceEnt.bf_de import Dice, DiceEnterprise, BernoulliFactory, Coin
from statsmodels.stats.proportion import proportion_confint
from typing import Union, List


def log_coeff_f_func(k: tuple):
    # Returns the value of the logarithm of the
    # ith coefficient of the series expansion for f(p1.p2)=p1/(1-p1*p2)
    if k[0] == k[1] + 1:
        return 0  # log(1)
    return -np.inf  # log(0)


def log_coeff_ci_func(i: int, t: Union[List, np.ndarray]):
    # Returns the value of the logarithm of ci, given by
    # ci = sum ak*t^k over k such that the sum is equal to i
    # In this case ci is not 0 only if i is odd, for which
    # there is only one ak = ((i-1)/2+1,(i-1)/2)
    if i % 2 == 1:
        return 0 + ((i - 1) / 2 + 1) * np.log(t[0]) + (i - 1) / 2 * np.log(t[1])
    return -np.inf


if __name__ == "__main__":
    np.random.seed(17)
    p1_seq = [0.1, np.sqrt(5)-2, 0.4, 0.3, 0.9]
    p2_seq = [0.2, np.sqrt(5)-2, 0.3, 0.6, 0.05]
    nobs = 10000
    test_mid_results = False

    for p1, p2 in [x for x in zip(p1_seq, p2_seq)]:
        p0 = 1 - p1 - p2
        if p0 <= 0:
            continue
        print(f"p1 = {p1}, p2 = {p2}")
        # Construct dice
        dice = Dice(p=[p0, p1, p2])
        # Define parameters
        fp = p1 / (1 - p1 * p2)
        eps = [1 - p1, 1 - p2]
        t = [1 - eps[0] / 2, 1 - eps[1] / 2]
        ft = (1 - eps[0] / 2) / (1 - (1 - eps[0] / 2) * (1 - eps[1] / 2))
        eps_p_t = [eps[0] / (2 - eps[0]), eps[1] / (2 - eps[1])]
        eps_fp_ft = 1 - (1-eps[0])/(1-(1-eps[0])*(1-eps[1]))
        if ft > 1 and fp > 1-eps_fp_ft:
            raise ValueError("Error in eps_fp_ft.")
        if ft <= 1:
            print(f"ft = {ft} <= 1")
            de_final = DiceEnterprise(
                dice=dice,
                type="positive_series",
                params={
                    "t": t,
                    "ft": ft,
                    "log_coeff_func": log_coeff_f_func,
                    "log_ci_func": lambda i: log_coeff_ci_func(i,t=t),
                    "eps": eps_p_t,
                },
            )
        else:
            print(f"ft = {ft} > 1. f(p) <= 1 - {eps_fp_ft:.6f} = {(1-eps_fp_ft):.6f}")
            de_series = DiceEnterprise(
                dice=dice,
                type="positive_series",
                params={
                    "t": t,
                    "ft": 1,
                    "log_coeff_func": lambda k: log_coeff_f_func(k) - np.log(ft),
                    "log_ci_func": lambda i: log_coeff_ci_func(i, t=t) - np.log(ft),
                    "eps": eps_p_t,
                },
            )
            if test_mid_results:
                dice.reset_num_rolls()
                res_test = de_series.sample(nobs)
                print(
                    f"Estimated value of f(p) = {np.mean(res_test)} (true value = {fp/ft:.6f}).\n# Tosses = {dice.get_num_rolls()}, Average tosses = {dice.get_num_rolls() / nobs}"
                )
                print(
                    proportion_confint(
                        count=np.sum(res_test), nobs=nobs, alpha=0.05, method="wilson"
                    )
                )
            de_final = BernoulliFactory(
                p1_coin=de_series,
                type="linear",
                params={"C": ft, "i": 1, "eps": eps_fp_ft},
            )
        dice.reset_num_rolls()
        res = de_final.sample(nobs)
        print(
            f"Estimated value of f(p) = {np.mean(res)} (true value = {fp:.6f}).\n# Tosses = {dice.get_num_rolls()}, Average tosses = {dice.get_num_rolls() / nobs}"
        )
        print(
            proportion_confint(
                count=np.sum(res), nobs=nobs, alpha=0.05, method="wilson"
            )
        )
        print("~~~~~~~~~~~~~~~~~~")
