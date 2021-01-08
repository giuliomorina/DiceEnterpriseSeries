import numpy as np
from DiceEnt.infinite_de import InfiniteDiceEnterprise


def coins_sampler(i):
    global p
    return 1 if np.random.rand() <= p * (1 - p) ** i else 0


def lower_bound(k):
    if k == 0:
        return 1.0 / 10
    return 9.0 / 10 * (1.0 / 10) ** k


def upper_bound(k):
    if k == 0:
        return 9.0 / 10
    elif k < 10:
        return float(k ** k) / ((k + 1) ** (k + 1))
    return 9.0 ** k / (10.0 ** (k + 1))


if __name__ == "__main__":
    np.random.seed(42)
    # True parameter of geometric
    p = 0.5
    # L1 norm of the lower bound
    lower_bound_norm = 1.0 / 5
    # L1 norm of the upper bound
    upper_bound_norm = (
        2.08898167698566409701515022108844664639140022914304338889010900332514
    )
    upper_bound_squared_norm = (
        0.93216278778354666929661251564329450900411663570521782920087617119190
    )
    lower_bound_dice = InfiniteDiceEnterprise(
        coins_sampler=coins_sampler,
        type="lower_bound",
        lower_bound_norm=lower_bound_norm,
        lower_bound_func=lower_bound,
    )
    accept_reject_dice = InfiniteDiceEnterprise(
        coins_sampler=coins_sampler,
        type="accept_reject",
        upper_bound_norm=upper_bound_norm,
        upper_bound_func=upper_bound,
    )
    upper_bound_dice = InfiniteDiceEnterprise(
        coins_sampler=coins_sampler,
        type="upper_bound",
        upper_bound_norm=upper_bound_norm,
        upper_bound_squared_norm=upper_bound_squared_norm,
        upper_bound_func=upper_bound,
    )
    res_upper_bound = upper_bound_dice.sample(1000)
    print(res_upper_bound)
    print(upper_bound_dice.get_num_tosses())

    res_accept_reject = accept_reject_dice.sample(1000)
    print(res_accept_reject)
    print(accept_reject_dice.get_num_tosses())