import numpy as np
from typing import Callable
from DiceEnt.bf_de import BernoulliFactory, Coin


class InfiniteDiceEnterprise:
    def __init__(
        self,
        type: str,
        coins_sampler: Callable,
        lower_bound_norm: float = None,
        lower_bound_func: Callable = None,
        upper_bound_norm: float = None,
        upper_bound_func: Callable = None,
        upper_bound_squared_norm: float = None,
    ):
        if type != "lower_bound" and type != "accept_reject" and type != "upper_bound":
            raise ValueError("Not valid type.")
        self._type = type
        self._coins_sampler = coins_sampler
        self._lower_bound_norm = lower_bound_norm
        self._lower_bound_func = lower_bound_func
        self._upper_bound_norm = upper_bound_norm
        self._upper_bound_squared_norm = upper_bound_squared_norm
        self._upper_bound_func = upper_bound_func
        self._num_tosses = 0

    def reset_num_tosses(self):
        self._num_tosses = 0
        return self

    def get_num_tosses(self):
        return self._num_tosses

    def _lower_bound_to_die(self):
        if self._lower_bound_norm is None or self._lower_bound_func is None:
            raise RuntimeError(
                "Specify both 'lower_bound_norm' and 'lower_bound_func'."
            )
        # We sequentially toss coins until we observe heads
        i = -1
        X = 0
        epsilon_sum = 0
        f0_coin = Coin(toss_coin_func=lambda: self._coins_sampler(0))
        coins = {}
        while X != 1:
            i += 1
            epsilon_sum += self._lower_bound_func(i)
            if i == 0:
                # Toss the first coin
                X = f0_coin.sample()
            else:
                coins[i - 1] = Coin(toss_coin_func=lambda: self._coins_sampler(i))
                # Need to toss a f_i/(1-sum^{i-1} f_j)
                # We resort to the division algorithm
                while True:
                    if np.random.rand() < 0.5:
                        # Toss an f_i-coin
                        W = coins[i - 1].sample()
                        if W == 1:
                            X = 1
                            break
                    else:
                        # Toss an 1-sum^{i} f_j, by tossing a
                        # sum^i f_j coin and then reversing the toss.
                        # By hypothesis we have the following bound:
                        # sum^i f_j <= 1-(|epsilon|-sum^i epsilon_j)
                        # Construct average coin
                        average_bf = BernoulliFactory(
                            p1_coin=f0_coin, type="average", params={"coins": coins}
                        )
                        # Construct sum coin
                        sum_bf = BernoulliFactory(
                            p1_coin=average_bf,
                            type="linear",
                            params={
                                "C": i + 1,
                                "i": 1,
                                "eps": self._lower_bound_norm - epsilon_sum,
                            },
                        )
                        # Construct reverse coin
                        reverse_bf = BernoulliFactory(p1_coin=sum_bf, type="reverse")
                        W = reverse_bf.sample()
                        if W == 1:
                            X = 0
                            break

        # Compute total number of tosses
        self._num_tosses += f0_coin.get_num_tosses()
        for coin in coins.values():
            self._num_tosses += coin.get_num_tosses()
        return i

    def _upper_bound_to_die(self):
        if (
            self._upper_bound_func is None
            or self._upper_bound_norm is None
            or self._upper_bound_squared_norm is None
        ):
            raise RuntimeError(
                "Define both 'upper_bound_func', 'upper_bound_norm' and 'upper_bound_squared_norm."
            )
        coins = {}
        res = None

        def log_J_coeff(i):
            return np.log(self._upper_bound_func(i)) - np.log(self._upper_bound_norm)

        while res is None:
            # Sample J
            J = BernoulliFactory.sample_categorical(log_J_coeff)
            # Prepare function to toss f_j*sum f_i*delta_i/c
            if J not in coins:
                coins[J] = Coin(toss_coin_func=lambda: self._coins_sampler(J))

            def toss_special_coin():
                if coins[J].sample() != 1:
                    return 0
                W = BernoulliFactory.sample_categorical(log_J_coeff)
                if W not in coins:
                    coins[W] = Coin(toss_coin_func=lambda: self._coins_sampler(W))
                return coins[W].sample()

            special_coin = Coin(toss_coin_func=toss_special_coin)
            bf = BernoulliFactory(
                p1_coin=special_coin,
                type="linear",
                params={
                    "C": self._upper_bound_norm / self._upper_bound_func(J),
                    "i": 1,
                    "eps": 1-self._upper_bound_squared_norm/self._upper_bound_norm,
                },
            )
            if bf.sample() == 1:
                res = J

        # Compute total number of tosses
        for coin in coins.values():
            self._num_tosses += coin.get_num_tosses()
        return res

    def _accept_reject(self):
        if self._upper_bound_func is None or self._upper_bound_norm is None:
            raise RuntimeError("Define both 'upper_bound_func' and 'upper_bound_norm'.")
        coins = {}
        res = None

        def log_K_coeff(i):
            return np.log(self._upper_bound_func(i)) - np.log(self._upper_bound_norm)

        while res is None:
            # Sample K
            K = BernoulliFactory.sample_categorical(log_K_coeff)
            # Construct coin
            if K not in coins:
                coins[K] = Coin(toss_coin_func=lambda: self._coins_sampler(K))
            # Sample 1/(2*delta_K)*f_K
            bf = BernoulliFactory(
                p1_coin=coins[K],
                type="linear",
                params={"C": 1.0 / (2 * self._upper_bound_func(K)), "i": 1, "eps": 0.5},
            )
            if bf.sample() == 1:
                res = K
        # Compute total number of tosses
        for coin in coins.values():
            self._num_tosses += coin.get_num_tosses()
        return res

    def sample(self, n: int = 1):
        if self._type == "lower_bound":
            func = self._lower_bound_to_die
        elif self._type == "accept_reject":
            func = self._accept_reject
        elif self._type == "upper_bound":
            func = self._upper_bound_to_die
        else:
            raise ValueError("Not valid type.")
        if n == 1:
            return func()
        else:
            return [func() for _ in np.arange(n)]
