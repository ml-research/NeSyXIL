#!/usr/bin/env python3

from typing import cast

import torch
from torch import Tensor

from captum.attr._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)

from .helpers.basic_models import (
    BasicModel,
    BasicModel6_MultiTensor,
    BasicModel_MultiLayer,
)
from .helpers.utils import BaseTest, assertArraysAlmostEqual


class Test(BaseTest):
    def test_apply_gradient_reqs(self) -> None:
        initial_grads = [False, True, False]
        test_tensor = torch.tensor([[6.0]], requires_grad=True)
        test_tensor.grad = torch.tensor([[7.0]])
        test_tensor_tuple = (torch.tensor([[5.0]]), test_tensor, torch.tensor([[7.0]]))
        out_mask = apply_gradient_requirements(test_tensor_tuple)
        for i in range(len(test_tensor_tuple)):
            self.assertTrue(test_tensor_tuple[i].requires_grad)
            self.assertEqual(out_mask[i], initial_grads[i])
            if test_tensor_tuple[i].grad is not None:
                self.assertAlmostEqual(
                    torch.sum(cast(Tensor, test_tensor_tuple[i].grad)).item(), 0.0
                )

    def test_undo_gradient_reqs(self) -> None:
        initial_grads = [False, True, False]
        test_tensor = torch.tensor([[6.0]], requires_grad=True)
        test_tensor.grad = torch.tensor([[7.0]])
        test_tensor_tuple = (
            torch.tensor([[6.0]], requires_grad=True),
            test_tensor,
            torch.tensor([[7.0]], requires_grad=True),
        )
        undo_gradient_requirements(test_tensor_tuple, initial_grads)
        for i in range(len(test_tensor_tuple)):
            self.assertEqual(test_tensor_tuple[i].requires_grad, initial_grads[i])
            if test_tensor_tuple[i].grad is not None:
                self.assertAlmostEqual(
                    torch.sum(cast(Tensor, test_tensor_tuple[i].grad)).item(), 0.0
                )

    def test_gradient_basic(self) -> None:
        model = BasicModel()
        input = torch.tensor([[5.0]], requires_grad=True)
        grads = compute_gradients(model, input)[0]
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [0.0], delta=0.01)

    def test_gradient_basic_2(self) -> None:
        model = BasicModel()
        input = torch.tensor([[-3.0]], requires_grad=True)
        grads = compute_gradients(model, input)[0]
        assertArraysAlmostEqual(grads.squeeze(0).tolist(), [1.0], delta=0.01)

    def test_gradient_multiinput(self) -> None:
        model = BasicModel6_MultiTensor()
        input1 = torch.tensor([[-3.0, -5.0]], requires_grad=True)
        input2 = torch.tensor([[-5.0, 2.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2))
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)
        assertArraysAlmostEqual(grads[1].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)

    def test_layer_gradient_linear0(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, -11.0, 23.0]], requires_grad=True)
        grads, eval, _ = compute_layer_gradients_and_eval(
            model, model.linear0, input, target_ind=0
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [4.0, 4.0, 4.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [5.0, -11.0, 23.0], delta=0.01
        )

    def test_layer_gradient_linear1(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval, _ = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )

    def test_layer_gradient_linear1_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval, is_layer_tuple = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )
        self.assertFalse(
            is_layer_tuple, ("Layer output should not be wrapped in " "a tuple.")
        )

    def test_layer_gradient_relu_input_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval, is_layer_tuple = compute_layer_gradients_and_eval(
            model, model.relu, input, target_ind=1, attribute_to_layer_input=True
        )
        assertArraysAlmostEqual(
            grads[0].squeeze(0).tolist(), [0.0, 1.0, 1.0, 1.0], delta=0.01
        )
        assertArraysAlmostEqual(
            eval[0].squeeze(0).tolist(), [-2.0, 9.0, 9.0, 9.0], delta=0.01
        )
        self.assertTrue(is_layer_tuple, "Layer input should be wrapped in a tuple.")

    def test_layer_gradient_output(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval, _ = compute_layer_gradients_and_eval(
            model, model.linear2, input, target_ind=1
        )
        assertArraysAlmostEqual(grads[0].squeeze(0).tolist(), [0.0, 1.0], delta=0.01)
        assertArraysAlmostEqual(eval[0].squeeze(0).tolist(), [26.0, 28.0], delta=0.01)
