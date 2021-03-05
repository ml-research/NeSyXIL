#!/usr/bin/env python3
import random

import numpy as np
import torch

from captum.attr import MSE, Max, Mean, Min, StdDev, Sum, Summarizer, Var

from .helpers.utils import BaseTest, assertArraysAlmostEqual, assertTensorAlmostEqual


def get_values(n=100, lo=None, hi=None, integers=False):
    for _ in range(n):
        if integers:
            yield random.randint(lo, hi)
        else:
            yield random.random() * (hi - lo) + lo


class Test(BaseTest):
    def test_div0(self):
        summarizer = Summarizer([Var(), Mean()])
        summ = summarizer.summary
        self.assertIsNone(summ)

        summarizer.update(torch.tensor(10))
        summ = summarizer.summary
        assertTensorAlmostEqual(self, summ["mean"], 10)
        assertTensorAlmostEqual(self, summ["variance"], 0)

        summarizer.update(torch.tensor(10))
        summ = summarizer.summary
        assertTensorAlmostEqual(self, summ["mean"], 10)
        assertTensorAlmostEqual(self, summ["variance"], 0)

    def test_var_defin(self):
        """
        Variance is avg squared distance to mean. Thus it should be positive.
        This test is to ensure this is the case.

        To test it, we will we make a skewed distribution leaning to one end
        (either very large or small values).

        We will also compare to numpy and ensure it is approximately the same.
        This is assuming numpy is correct, for which it should be.
        """
        SMALL_VAL = -10000
        BIG_VAL = 10000
        AMOUNT_OF_SMALLS = [100, 10]
        AMOUNT_OF_BIGS = [10, 100]
        for sm, big in zip(AMOUNT_OF_SMALLS, AMOUNT_OF_BIGS):
            summ = Summarizer([Var()])
            values = []
            for _ in range(sm):
                values.append(SMALL_VAL)
                summ.update(torch.tensor(SMALL_VAL, dtype=torch.float64))

            for _ in range(big):
                values.append(BIG_VAL)
                summ.update(torch.tensor(BIG_VAL, dtype=torch.float64))

            actual_var = np.var(values)
            actual_var = torch.from_numpy(np.array(actual_var))

            var = summ.summary["variance"]

            assertTensorAlmostEqual(self, var, actual_var)
            self.assertTrue((var > 0).all())

    def test_multi_dim(self):
        x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x2 = torch.tensor([2.0, 1.0, 2.0, 4.0])
        x3 = torch.tensor([3.0, 3.0, 1.0, 4.0])

        summarizer = Summarizer([Mean(), Var()])
        summarizer.update(x1)
        assertArraysAlmostEqual(summarizer.summary["mean"], x1)
        assertArraysAlmostEqual(summarizer.summary["variance"], torch.zeros_like(x1))

        summarizer.update(x2)
        assertArraysAlmostEqual(
            summarizer.summary["mean"], torch.tensor([1.5, 1.5, 2.5, 4])
        )
        assertArraysAlmostEqual(
            summarizer.summary["variance"], torch.tensor([0.25, 0.25, 0.25, 0])
        )

        summarizer.update(x3)
        assertArraysAlmostEqual(summarizer.summary["mean"], torch.tensor([2, 2, 2, 4]))
        assertArraysAlmostEqual(
            summarizer.summary["variance"],
            torch.tensor([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0]),
        )

    def test_stats_random_data(self):
        N = 1000
        BIG_VAL = 100000
        values = list(get_values(lo=-BIG_VAL, hi=BIG_VAL, n=N))
        stats_to_test = [
            Mean(),
            Var(),
            Var(order=1),
            StdDev(),
            StdDev(order=1),
            Min(),
            Max(),
            Sum(),
            MSE(),
        ]
        stat_names = [
            "mean",
            "variance",
            "sample_variance",
            "std_dev",
            "sample_std_dev",
            "min",
            "max",
            "sum",
            "mse",
        ]
        gt_fns = [
            np.mean,
            np.var,
            lambda x: np.var(x, ddof=1),
            np.std,
            lambda x: np.std(x, ddof=1),
            np.min,
            np.max,
            np.sum,
            lambda x: np.sum((x - np.mean(x)) ** 2),
        ]

        for stat, name, gt in zip(stats_to_test, stat_names, gt_fns):
            summ = Summarizer([stat])
            for x in values:
                summ.update(torch.tensor(x, dtype=torch.float64))

            actual = torch.from_numpy(np.array(gt(values)))
            stat_val = summ.summary[name]

            # rounding errors is a serious issue (moreso for MSE)
            assertTensorAlmostEqual(self, stat_val, actual, delta=0.005)
