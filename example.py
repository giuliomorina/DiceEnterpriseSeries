import numpy as np
from DiceEnt.bf import Coin, BernoulliFactory
from scipy.special import gammaln
from statsmodels.stats.proportion import proportion_confint

def log_binomial(n: int, k: int):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def log_coeff_f_func(i: int):
    # Returns the value of the logarithm of the absolute value of the
    # ith coefficient of the series expansion for f(p)=p*sqrt(2-p)
    if i <= 0:
        return -np.inf
    if i == 1:
        return 0.5 * np.log(2)
    return (
        -0.5 * np.log(2)
        + (5 - 3 * i) * np.log(2)
        - np.log(i - 1)
        + log_binomial(2 * i - 4, i - 2)
    )


def log_coeff_h_func(i: int):
    # Returns the value of the logarithm of the ith component
    # of the series expansion of h(p)=sqrt(2)p-p*sqrt(2-p)
    if i <= 1:
        return -np.inf
    else:
        return log_coeff_f_func(i)


def sample_g_M(p_coin: Coin, M: float):
    # Samples from g(p)/M = sqrt(2)/M*p coin
    if M < np.sqrt(2):
        raise ValueError("M should be greater than sqrt(2)")
    return 0 if np.random.rand() > np.sqrt(2) / M else p_coin.sample()


def log_coeff_h_M_func(i: int):
    return log_coeff_h_func(i) - np.log(M)

if __name__ == "__main__":
    np.random.seed(42)
    p_seq = [0.01,0.25, 0.5, 0.75, 0.99]
    M = 2 * np.sqrt(2) - 1
    t = 1
    ft = (np.sqrt(2) * t - t * np.sqrt(2 - t)) / M
    nobs = 1000
    test_mid_results = False

    for p in p_seq:
        print(f"p = {p}")
        true_g_M_val = np.sqrt(2) / M * p
        true_h_M_val = (np.sqrt(2) * p - p * np.sqrt(2 - p)) / M
        true_f_M_val = p * np.sqrt(2 - p) / M

        def toss_coin_func():
            return 1 if np.random.rand() <= p else 0
        p_coin = Coin(toss_func=toss_coin_func)
        # Construct Bernoulli Factory for g(p)/M = sqrt(2)/M*p
        bf_g_M = BernoulliFactory(
            p1_coin=p_coin, type="linear", params={"C": np.sqrt(2) / M, "eps": None, "i": 1}
        )
        # Construct Bernoulli Factory for h(p)/M
        bf_h_M = BernoulliFactory(
            p1_coin=p_coin,
            type="positive_series",
            params={"t": t, "ft": ft, "log_coeff_func": log_coeff_h_M_func, "eps": None},
        )
        # Construct Bernoulli Factory for f(p)/M = (g(p)-h(p))/M using the best possible epsilon (minimise # tosses)
        bf_f_M = BernoulliFactory(
            p1_coin=bf_g_M, type="difference", params={"p0_coin": bf_h_M, "eps": true_g_M_val - true_h_M_val}
        )
        # Construct Bernoulli Factory for f(p) using the best possible epsilon
        bf_f = BernoulliFactory(p1_coin=bf_f_M, type="linear", params={"C": M, "i": 1, "eps": 1 - true_f_M_val * M})
        if test_mid_results:
            # Test that it samples correctly according to g(p)/M
            test_g_M_res = bf_g_M.sample(nobs)
            print(
                f"Estimated value of g(p)/M = {np.mean(test_g_M_res)} (true value = {true_g_M_val}). # Tosses = {p_coin.get_num_tosses()}"
            )
            p_coin.reset_num_tosses()
            # Test that it samples correctly according to h(p)/M
            test_h_M_res = bf_h_M.sample(nobs)
            print(
                f"Estimated value of h(p)/M = {np.mean(test_h_M_res)} (true value = {true_h_M_val}). # Tosses = {p_coin.get_num_tosses()}"
            )
            p_coin.reset_num_tosses()
            # Test that it samples correctly according to f(p)/M = (g(p)-h(p))/M
            test_f_M_res = bf_f_M.sample(nobs)
            print(
                f"Estimated value of f(p)/M = {np.mean(test_f_M_res)} (true value = {true_f_M_val}). # Tosses = {p_coin.get_num_tosses()}"
            )
            p_coin.reset_num_tosses()

        test_f_res = bf_f.sample(nobs)
        print(
            f"Estimated value of f(p) = {np.mean(test_f_res)} (true value = {true_f_M_val*M}). # Tosses = {p_coin.get_num_tosses()}, Average tosses = {p_coin.get_num_tosses()/nobs}"
        )
        p_coin.reset_num_tosses()
        # Compute 95% confidence interval using a normal approximation
        print(proportion_confint(count=np.sum(test_f_res), nobs=nobs, alpha=0.05, method="normal"))