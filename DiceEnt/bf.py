import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Union
from collections import deque


class Coin:
    def __init__(self, toss_func: Callable):
        self._toss_func = toss_func
        self._num_tosses = 0

    def sample(self):
        self._num_tosses += 1
        return self._toss_func()

    def reset_num_tosses(self):
        self._num_tosses = 0
        return self

    def get_num_tosses(self):
        return self._num_tosses


class BernoulliFactory:
    def __init__(
        self,
        p1_coin: Union[Coin, "BernoulliFactory"],
        type: str,
        params: Dict[str, Any] = None,
    ):
        self._check_type(type, params)
        self._p1_coin = p1_coin
        self._type = type
        self._params = None
        self.set_params(params)

    def _check_type(self, type: str, params: Dict[str, Any]):
        if type == "logistic":
            if "C" not in params:
                raise ValueError("Specify 'C' for a logistic Bernoulli Factory.")
        elif type == "linear":
            if "C" not in params or "eps" not in params or "i" not in params:
                raise ValueError(
                    "Specify 'C', 'eps', 'i' for a linear Bernoulli Factory."
                )
        elif type == "difference":
            if "p0_coin" not in params or "eps" not in params:
                raise ValueError(
                    "Specify 'p0_coin', 'eps' for a difference Bernoulli Factory."
                )
        elif type == "average":
            if "p0_coin" not in params:
                raise ValueError("Specify 'p0_coin' for an average Bernoulli Factory.")
        elif type == "reverse":
            return self
        elif type == "positive_series":
            if (
                "t" not in params
                or "ft" not in params
                or "log_coeff_func" not in params
                or "eps" not in params
            ):
                raise ValueError(
                    "Specify 't', 'ft', 'log_coeff_func, 'eps' for a positive power series Bernoulli Factory."
                )
        else:
            raise ValueError("Type not recognised.")
        return self

    def _logistic(self):
        # f(p) = Cp/(1+Cp)
        # The function is taken from "Designing perfect simulation algorithms
        # using local correctness" by M. Huber (2019), in an iterative form rather than
        # recursive. This is equivalent to the 2-coin algorithm proposed in
        # "Barker's algorithm for Bayesian inference with intractable likelihoods"
        # by F. Goncalves et al. (2017)
        C = self._params["C"]
        while True:
            if np.random.rand() > C / (1 + C):
                return 0
            X = self._p1_coin.sample()
            if X == 1:
                return 1

    def _linear(self):
        # f(p) = (C*p)^i, assuming C*p<1-eps
        # The function is taken from "Designing perfect simulation algorithms
        # using local correctness" by M. Huber (2019), in an iterative form rather than
        # recursive which uses a custom stack, based on:
        # https://www.codeproject.com/Articles/418776/How-to-replace-recursive-functions-using-stack-and
        if self._params["C"] <= 1:
            res = 1
            counter = 1
            while res == 1 and counter <= self._params["i"]:
                if np.random.rand() > self._params["C"]:
                    return 0
                else:
                    res = 0 if self._p1_coin.sample() != 1 else 1
                counter += 1
            return res
        # Initialise logistic bf
        logistic_bf = BernoulliFactory(
            p1_coin=self._p1_coin, type="logistic", params={"C": -9}
        )
        # Initialise stack and return value
        return_value = -1
        stack = deque()
        stack.append(
            {
                "C": self._params["C"],
                "i": self._params["i"],
                "eps": self._params["eps"],
                "stage": 0,
            }
        )
        while len(stack) > 0:
            current_stack = stack.pop()
            if current_stack["stage"] == 0:
                if current_stack["i"] == 0:
                    return_value = 1
                elif current_stack["i"] > 3.55 / current_stack["eps"]:
                    beta = (1.0 - current_stack["eps"] / 2.0) / (
                        1.0 - current_stack["eps"]
                    )
                    # Draw B1 as Bernoulli beta^-i
                    B1 = 1 if np.random.rand() <= beta ** (-current_stack["i"]) else 0
                    if B1 == 0:
                        return_value = 0
                    else:
                        # Push to stack
                        current_stack["stage"] = 1
                        stack.append(current_stack)
                        stack.append(
                            {
                                "C": beta * current_stack["C"],
                                "i": current_stack["i"],
                                "eps": current_stack["eps"],
                                "stage": 0,
                            }
                        )
                else:
                    # Draw B2 from the logistic BF
                    logistic_bf.set_params({"C": current_stack["C"]})
                    B2 = logistic_bf.sample()
                    # Push to the stack
                    current_stack["stage"] = 2
                    stack.append(current_stack)
                    stack.append(
                        {
                            "C": current_stack["C"],
                            "i": current_stack["i"] + 1 - 2 * B2,
                            "eps": current_stack["eps"],
                            "stage": 0,
                        }
                    )
            elif current_stack["stage"] == 1:
                # This would contain things to do after the 1st recursive call
                # in this case there is none.
                continue
            elif current_stack["stage"] == 2:
                # This would contain things to do after the 2nd recursive call
                # in this case there is none.
                continue
        return return_value

    def _difference(self):
        # f(p0,p1) = p1-p0, assuming pi-p0 >= eps
        p0_coin = self._params["p0_coin"]
        eps = self._params["eps"]
        p1_reverse_coin = BernoulliFactory(p1_coin=self._p1_coin, type="reverse")
        average_bf = BernoulliFactory(
            p1_coin=p1_reverse_coin, type="average", params={"p0_coin": p0_coin}
        )
        double_bf = BernoulliFactory(
            p1_coin=average_bf, type="linear", params={"C": 2, "i": 1, "eps": eps}
        )
        reverse_bf = BernoulliFactory(p1_coin=double_bf, type="reverse")
        return reverse_bf.sample()

    def _average(self):
        # f(p0,p1) = (p1+p0)/2
        return (
            self._p1_coin.sample()
            if np.random.rand() < 0.5
            else self._params["p0_coin"].sample()
        )

    def _reverse(self):
        return 1 if self._p1_coin.sample() != 1 else 0

    def _positive_series(self):
        t = self._params["t"]
        ft = self._params["ft"]
        # Return 0 with probability 1-f(t)
        if np.random.rand() > ft:
            return 0
        # Sample K proportional to a_kt^k/f(t)
        def K_log_coeff_func(k):
            return self._params["log_coeff_func"](k) + k * np.log(t) - np.log(ft)

        K = self.sample_categorical(K_log_coeff_func)
        if K == 0:
            return 1
        # Toss a (p/t)^k-coin
        linear_bf = BernoulliFactory(
            p1_coin=self._p1_coin,
            type="linear",
            params={"C": 1.0 / t, "i": K, "eps": self._params["eps"]},
        )
        return linear_bf.sample()

    @staticmethod
    def sample_categorical(log_coeff_func):
        log_U = np.log(np.random.rand())
        log_S = -np.inf
        i = 0
        while True:
            log_S = np.logaddexp(log_S, log_coeff_func(i))
            if log_U < log_S:
                return i
            i += 1

    def sample(self, n: int = 1):
        if self._type == "logistic":
            func = self._logistic
        elif self._type == "linear":
            func = self._linear
        elif self._type == "difference":
            func = self._difference
        elif self._type == "average":
            func = self._average
        elif self._type == "reverse":
            func = self._reverse
        elif self._type == "positive_series":
            func = self._positive_series
        else:
            raise RuntimeError("Not valid 'type'.")
        if n == 1:
            return func()
        else:
            return [func() for _ in np.arange(n)]

    def set_params(self, params):
        self._params = params
        return self
