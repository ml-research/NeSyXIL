#!/usr/bin/env python3

import unittest
from typing import Any, List, Tuple, Union, cast

import torch
from torch import Tensor
from torch.nn import Module

from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._utils.typing import BaselineType

from ..helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from ..helpers.conductance_reference import ConductanceReference
from ..helpers.utils import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorTuplesAlmostEqual,
)


class Test(BaseTest):
    def test_simple_input_conductance(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_test_assert(net, net.linear0, inp, [[0.0, 390.0, 0.0]])

    def test_simple_input_multi_conductance(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_test_assert(
            net,
            net.relu,
            inp,
            ([[90.0, 100.0, 100.0, 100.0]], [[90.0, 100.0, 100.0, 100.0]]),
        )

    def test_simple_input_with_scalar_baseline_conductance(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_test_assert(
            net, net.linear0, inp, [[0.0, 390.0, 0.0]], baselines=0.0
        )

    def test_simple_linear_conductance(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_test_assert(
            net, net.linear1, inp, [[90.0, 100.0, 100.0, 100.0]]
        )

    def test_simple_relu_conductance(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_test_assert(net, net.relu, inp, [[90.0, 100.0, 100.0, 100.0]])

    def test_simple_output_conductance(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_test_assert(net, net.linear2, inp, [[390.0, 0.0]])

    def test_simple_multi_input_linear2_conductance(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._conductance_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            [[390.0, 0.0]],
            additional_args=(4,),
        )

    def test_simple_multi_input_relu_conductance(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._conductance_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            [[90.0, 100.0, 100.0, 100.0]],
            additional_args=(inp3, 5),
        )

    def test_simple_multi_input_relu_conductance_batch(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0], [0.0, 0.0, 10.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0], [0.0, 0.0, 10.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        self._conductance_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            [[90.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0]],
            additional_args=(inp3, 5),
        )

    def test_matching_conv1_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.conv1, inp)

    def test_matching_pool1_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10)
        self._conductance_reference_test_assert(net, net.pool1, inp)

    def test_matching_conv2_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.conv2, inp)

    def test_matching_pool2_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10)
        self._conductance_reference_test_assert(net, net.pool2, inp)

    def test_matching_conv_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(4, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.relu3, inp)

    def test_matching_conv_with_baseline_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10)
        baseline = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_reference_test_assert(net, net.fc1, inp, baseline)

    def _conductance_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_conductance: Union[List[List[float]], Tuple[List[List[float]], ...]],
        baselines: BaselineType = None,
        additional_args: Any = None,
    ) -> None:
        cond = LayerConductance(model, target_layer)
        for internal_batch_size in (None, 1, 20):
            attributions, delta = cond.attribute(
                test_input,
                baselines=baselines,
                target=0,
                n_steps=500,
                method="gausslegendre",
                additional_forward_args=additional_args,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )
            delta_condition = all(abs(delta.numpy().flatten()) < 0.01)
            self.assertTrue(
                delta_condition,
                "Sum of attributions does {}"
                " not match the difference of endpoints.".format(delta),
            )

            assertTensorTuplesAlmostEqual(
                self, attributions, expected_conductance, delta=0.1,
            )

    def _conductance_reference_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: Tensor,
        test_baseline: Union[None, Tensor] = None,
    ) -> None:
        layer_output = None

        def forward_hook(module, inp, out):
            nonlocal layer_output
            layer_output = out

        hook = target_layer.register_forward_hook(forward_hook)
        final_output = model(test_input)
        layer_output = cast(Tensor, layer_output)
        hook.remove()
        target_index = torch.argmax(torch.sum(final_output, 0))
        cond = LayerConductance(model, target_layer)
        cond_ref = ConductanceReference(model, target_layer)
        attributions, delta = cast(
            Tuple[Tensor, Tensor],
            cond.attribute(
                test_input,
                baselines=test_baseline,
                target=target_index,
                n_steps=300,
                method="gausslegendre",
                return_convergence_delta=True,
            ),
        )
        delta_condition = all(abs(delta.numpy().flatten()) < 0.005)
        self.assertTrue(
            delta_condition,
            "Sum of attribution values does {} "
            " not match the difference of endpoints.".format(delta),
        )

        attributions_reference = cond_ref.attribute(
            test_input,
            baselines=test_baseline,
            target=target_index,
            n_steps=300,
            method="gausslegendre",
        )

        # Check that layer output size matches conductance size.
        self.assertEqual(layer_output.shape, attributions.shape)
        # Check that reference implementation output matches standard implementation.
        assertArraysAlmostEqual(
            attributions.reshape(-1).tolist(),
            attributions_reference.reshape(-1).tolist(),
            delta=0.07,
        )

        # Test if batching is working correctly for inputs with multiple examples
        if test_input.shape[0] > 1:
            for i in range(test_input.shape[0]):
                single_attributions = cast(
                    Tensor,
                    cond.attribute(
                        test_input[i : i + 1],
                        baselines=test_baseline[i : i + 1]
                        if test_baseline is not None
                        else None,
                        target=target_index,
                        n_steps=300,
                        method="gausslegendre",
                    ),
                )
                # Verify that attributions when passing example independently
                # matches corresponding attribution of batched input.
                assertArraysAlmostEqual(
                    attributions[i : i + 1].reshape(-1).tolist(),
                    single_attributions.reshape(-1).tolist(),
                    delta=0.01,
                )


if __name__ == "__main__":
    unittest.main()
