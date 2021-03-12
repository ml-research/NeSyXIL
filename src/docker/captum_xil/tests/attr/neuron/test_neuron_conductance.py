#!/usr/bin/env python3

import unittest
from typing import Any, List, Tuple, Union, cast

import torch
from torch import Tensor
from torch.nn import Module

from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.neuron.neuron_conductance import NeuronConductance
from captum.attr._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric

from ..helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from ..helpers.utils import BaseTest, assertArraysAlmostEqual


class Test(BaseTest):
    def test_simple_conductance_input_linear2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_input_test_assert(
            net, net.linear2, inp, (0,), [0.0, 390.0, 0.0]
        )

    def test_simple_conductance_input_linear1(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_input_test_assert(net, net.linear1, inp, 0, [0.0, 90.0, 0.0])

    def test_simple_conductance_input_relu(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 70.0, 30.0]], requires_grad=True)
        self._conductance_input_test_assert(net, net.relu, inp, (3,), [0.0, 70.0, 30.0])

    def test_simple_conductance_multi_input_linear2(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._conductance_input_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            (0,),
            ([[0.0, 156.0, 0.0]], [[0.0, 156.0, 0.0]], [[0.0, 78.0, 0.0]]),
            (4,),
        )

    def test_simple_conductance_multi_input_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._conductance_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (3,),
            ([[0.0, 50.0, 5.0]], [[0.0, 20.0, 25.0]]),
            (inp3, 5),
        )

    def test_simple_conductance_multi_input_batch_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0], [0.0, 0.0, 10.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0], [0.0, 0.0, 10.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        self._conductance_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (3,),
            (
                [[0.0, 50.0, 5.0], [0.0, 0.0, 50.0]],
                [[0.0, 20.0, 25.0], [0.0, 0.0, 50.0]],
            ),
            (inp3, 5),
        )

    def test_matching_conv2_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10)
        self._conductance_input_sum_test_assert(net, net.conv2, inp, 0.0)

        # trying different baseline
        self._conductance_input_sum_test_assert(net, net.conv2, inp, 0.000001)

    def test_matching_relu2_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.relu2, inp, baseline)

    def test_matching_relu2_with_scalar_base_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.relu2, inp, 0.0)

    def test_matching_pool2_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10)
        baseline = 20 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.pool2, inp, baseline)

    def _conductance_input_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        test_neuron: Union[int, Tuple[int, ...]],
        expected_input_conductance: Union[List[float], Tuple[List[List[float]], ...]],
        additional_input: Any = None,
    ) -> None:
        for internal_batch_size in (None, 1, 20):
            cond = NeuronConductance(model, target_layer)
            attributions = cond.attribute(
                test_input,
                test_neuron,
                target=0,
                n_steps=500,
                method="gausslegendre",
                additional_forward_args=additional_input,
                internal_batch_size=internal_batch_size,
            )
            if isinstance(expected_input_conductance, tuple):
                for i in range(len(expected_input_conductance)):
                    for j in range(len(expected_input_conductance[i])):
                        assertArraysAlmostEqual(
                            attributions[i][j : j + 1].squeeze(0).tolist(),
                            expected_input_conductance[i][j],
                            delta=0.1,
                        )
            else:
                if isinstance(attributions, Tensor):
                    assertArraysAlmostEqual(
                        attributions.squeeze(0).tolist(),
                        expected_input_conductance,
                        delta=0.1,
                    )
                else:
                    raise AssertionError(
                        "Attributions not returning a Tensor when expected."
                    )

    def _conductance_input_sum_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        test_baseline: BaselineType = None,
    ):
        layer_cond = LayerConductance(model, target_layer)
        attributions = cast(
            Tensor,
            layer_cond.attribute(
                test_input,
                baselines=test_baseline,
                target=0,
                n_steps=500,
                method="gausslegendre",
            ),
        )
        neuron_cond = NeuronConductance(model, target_layer)
        attr_shape = cast(Tuple[int, ...], attributions.shape)
        for i in range(attr_shape[1]):
            for j in range(attr_shape[2]):
                for k in range(attr_shape[3]):
                    neuron_vals = neuron_cond.attribute(
                        test_input,
                        (i, j, k),
                        baselines=test_baseline,
                        target=0,
                        n_steps=500,
                    )
                    for n in range(attributions.shape[0]):
                        self.assertAlmostEqual(
                            torch.sum(neuron_vals[n]).item(),
                            attributions[n, i, j, k].item(),
                            delta=0.005,
                        )


if __name__ == "__main__":
    unittest.main()
