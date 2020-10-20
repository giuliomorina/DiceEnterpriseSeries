import numpy as np
import pandas as pd
from DiceEnt.bf import Coin, BernoulliFactory


def sample_table(n: int, bf: BernoulliFactory, p1_coin: Coin, p0_coin: Coin):
    res_dict = {i: None for i in np.arange(1, n + 1)}
    for i in np.arange(1, n + 1):
        p1_coin.reset_num_tosses()
        p0_coin.reset_num_tosses()
        res_dict[i] = {
            "res": bf.sample(),
            "num_tosses_p1": p1_coin.get_num_tosses(),
            "num_tosses_p0": p0_coin.get_num_tosses(),
        }
    return pd.DataFrame(res_dict)


def test_bf():
    np.random.seed(17)
    n = 10000
    p1 = 0.8
    p0 = 0.3

    def toss_p1_coin():
        return 1 if np.random.rand() < p1 else 0

    def toss_p0_coin():
        return 1 if np.random.rand() < p0 else 0

    p1_coin = Coin(toss_func=toss_p1_coin)
    p0_coin = Coin(toss_func=toss_p0_coin)
    # Test reverse
    reverse_bf = BernoulliFactory(p1_coin, "reverse")
    reverse_bf_res = sample_table(n=n,bf=reverse_bf,p1_coin=p1_coin,p0_coin=p0_coin)
    assert np.allclose(reverse_bf_res.mean(axis=1)["res"], 1-p1, atol=0.01)
    assert reverse_bf_res.mean(axis=1)["num_tosses_p1"] == 1
    # Test average
    average_bf = BernoulliFactory(p1_coin, "average", {"p0_coin": p0_coin})
    average_bf_res = sample_table(n=n, bf=average_bf, p1_coin=p1_coin, p0_coin=p0_coin)
    assert np.allclose(average_bf_res.mean(axis=1)["res"], (p0+p1)/2, atol=0.01)
    assert np.allclose(average_bf_res.mean(axis=1)["num_tosses_p1"], 0.5, atol=0.01)
    assert np.allclose(average_bf_res.mean(axis=1)["num_tosses_p0"], 0.5, atol=0.01)
    # Test doubling
    doubling_bf = BernoulliFactory(p0_coin, "linear", {"C": 2, "eps":0.4, "i": 1})
    doubling_bf_res = sample_table(n=n, bf=doubling_bf, p1_coin=p1_coin, p0_coin=p0_coin)
    assert np.allclose(doubling_bf_res.mean(axis=1)["res"], 2*p0, atol=0.01)
    # Test difference
    differenc_bf = BernoulliFactory(p1_coin, "difference", {"eps": 0.3, "p0_coin": p0_coin})
    differenc_bf_res = sample_table(n=n, bf=differenc_bf, p1_coin=p1_coin, p0_coin=p0_coin)
    assert np.allclose(differenc_bf_res.mean(axis=1)["res"], p1 - p0, atol=0.01)