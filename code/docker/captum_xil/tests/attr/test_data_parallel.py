#!/usr/bin/env python3

import unittest

import torch

from captum.attr._core.deep_lift import DeepLift
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.neuron.neuron_conductance import NeuronConductance
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from captum.attr._core.neuron.neuron_feature_ablation import NeuronFeatureAblation
from captum.attr._core.neuron.neuron_gradient import NeuronGradient
from captum.attr._core.neuron.neuron_gradient_shap import NeuronGradientShap
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)
from captum.attr._core.occlusion import Occlusion
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling

from .helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    ReLULinearDeepLiftModel,
)
from .helpers.utils import BaseGPUTest


class Test(BaseGPUTest):
    def test_simple_input_internal_inf(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            InternalInfluence, net, net.relu, inputs=inp, target=0
        )

    def test_multi_output_internal_inf(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            InternalInfluence, net, net.relu, inputs=inp, target=0
        )

    def test_multi_input_internal_inf(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            InternalInfluence,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=0,
            test_batches=True,
        )

    def test_simple_layer_activation(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(LayerActivation, net, net.relu, inputs=inp)

    def test_multi_output_layer_activation(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerActivation, net, net.relu, alt_device_ids=True, inputs=inp
        )

    def test_multi_input_layer_activation(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerActivation,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
        )

    def test_simple_layer_conductance(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerConductance, net, net.relu, inputs=inp, target=1
        )

    def test_multi_output_layer_conductance(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerConductance, net, net.relu, alt_device_ids=True, inputs=inp, target=1
        )

    def test_multi_input_layer_conductance(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerConductance,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
            test_batches=True,
        )

    def test_multi_dim_layer_conductance(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerConductance, net, net.conv2, alt_device_ids=True, inputs=inp, target=1
        )

    def test_simple_layer_integrated_gradients(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerIntegratedGradients, net, net.relu, inputs=inp, target=1
        )

    def test_multi_input_layer_integrated_gradients(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerIntegratedGradients,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
            test_batches=True,
        )

    def test_multi_dim_layer_integrated_gradients(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerIntegratedGradients,
            net,
            net.conv2,
            alt_device_ids=True,
            inputs=inp,
            target=1,
        )

    def test_multi_output_layer_integrated_gradients(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerIntegratedGradients,
            net,
            net.relu,
            alt_device_ids=True,
            inputs=inp,
            target=1,
        )

    def test_simple_layer_gradient_x_activation(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerGradientXActivation, net, net.relu, inputs=inp, target=1
        )

    def test_multi_output_layer_gradient_x_activation(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerGradientXActivation,
            net,
            net.relu,
            alt_device_ids=True,
            inputs=inp,
            target=1,
        )

    def test_multi_input_layer_gradient_x_activation(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            LayerGradientXActivation,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
        )

    def test_multi_dim_layer_grad_cam(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            LayerGradCam, net, net.conv2, alt_device_ids=True, inputs=inp, target=1
        )

    def test_multi_output_layer_grad_cam(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            LayerGradCam, net, net.relu, alt_device_ids=True, inputs=inp, target=1
        )

    def test_multi_input_layer_ablation(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        for perturbations_per_eval in [1, 2, 3]:
            self._data_parallel_test_assert(
                LayerFeatureAblation,
                net,
                net.model.relu,
                alt_device_ids=False,
                inputs=(inp1, inp2),
                additional_forward_args=(inp3, 5),
                target=1,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_multi_output_layer_ablation(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        for perturbations_per_eval in [1, 2, 3]:
            self._data_parallel_test_assert(
                LayerFeatureAblation,
                net,
                net.relu,
                alt_device_ids=False,
                inputs=inp,
                target=1,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_multi_dim_layer_ablation(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                LayerFeatureAblation,
                net,
                net.conv2,
                alt_device_ids=True,
                inputs=inp,
                target=1,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_simple_neuron_conductance(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            NeuronConductance, net, net.relu, inputs=inp, neuron_index=3, target=1
        )

    def test_multi_input_neuron_conductance(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronConductance,
            net,
            net.model.relu,
            alt_device_ids=True,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            target=1,
            neuron_index=(3,),
            test_batches=True,
        )

    def test_multi_dim_neuron_conductance(self):
        net = BasicModel_ConvNet().cuda()
        inp = 100 * torch.randn(4, 1, 10, 10).cuda()
        self._data_parallel_test_assert(
            NeuronConductance,
            net,
            net.conv2,
            alt_device_ids=True,
            inputs=inp,
            target=1,
            neuron_index=(0, 1, 0),
        )

    def test_simple_neuron_gradient(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            NeuronGradient,
            net,
            net.relu,
            alt_device_ids=True,
            inputs=inp,
            neuron_index=3,
        )

    def test_multi_input_neuron_gradient(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronGradient,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
        )

    def test_simple_neuron_integrated_gradient(self):
        net = BasicModel_MultiLayer().cuda()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [20.0, 100.0, 120.0],
                [30.0, 10.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ).cuda()
        self._data_parallel_test_assert(
            NeuronIntegratedGradients,
            net,
            net.relu,
            alt_device_ids=True,
            inputs=inp,
            neuron_index=3,
        )

    def test_multi_input_integrated_gradient(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronIntegratedGradients,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
            test_batches=True,
        )

    def test_multi_input_guided_backprop(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronGuidedBackprop,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
        )

    def test_multi_input_deconv(self):
        net = BasicModel_MultiLayer_MultiInput().cuda()
        inp1, inp2, inp3 = (
            10 * torch.randn(12, 3).cuda(),
            5 * torch.randn(12, 3).cuda(),
            2 * torch.randn(12, 3).cuda(),
        )
        self._data_parallel_test_assert(
            NeuronDeconvolution,
            net,
            net.model.relu,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3, 5),
            neuron_index=(3,),
        )

    def test_multi_input_deeplift(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor(
            [[-10.0, 1.0, -5.0], [1.9, 2.0, 1.9]], requires_grad=True
        ).cuda()
        inp2 = torch.tensor(
            [[3.0, 3.0, 1.0], [1.2, 3.0, 2.3]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            DeepLift,
            net,
            None,
            inputs=(inp1, inp2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_layer_deeplift(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor(
            [[-10.0, 1.0, -5.0], [1.0, 2.0, 3.0]], requires_grad=True
        ).cuda()
        inp2 = torch.tensor(
            [[3.0, 3.0, 1.0], [4.5, 6.3, 2.3]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            LayerDeepLift,
            net,
            net.l3,
            inputs=(inp1, inp2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_layer_deeplift_shap(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()
        base2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            LayerDeepLiftShap,
            net,
            net.l3,
            inputs=(inp1, inp2),
            baselines=(base1, base2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_output_layer_deeplift_shap(self):
        net = BasicModel_MultiLayer(multi_input_module=True).cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()
        base2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            LayerDeepLiftShap,
            net,
            net.relu,
            inputs=(inp1, inp2),
            target=0,
            baselines=(base1, base2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_neuron_deeplift(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        self._data_parallel_test_assert(
            NeuronDeepLift,
            net,
            net.l3,
            inputs=(inp1, inp2),
            neuron_index=0,
            additional_forward_args=None,
            test_batches=False,
        )

    def test_multi_input_neuron_deeplift_shap(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()
        base2 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ).cuda()

        self._data_parallel_test_assert(
            NeuronDeepLiftShap,
            net,
            net.l3,
            inputs=(inp1, inp2),
            neuron_index=0,
            baselines=(base1, base2),
            additional_forward_args=None,
            test_batches=False,
        )

    def test_basic_gradient_shap_helper(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, GradientShap, None)

    def test_basic_gradient_shap_helper_with_alt_devices(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, GradientShap, None, True)

    def test_basic_neuron_gradient_shap(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, NeuronGradientShap, net.linear2, False)

    def test_basic_neuron_gradient_shap_with_alt_devices(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(net, NeuronGradientShap, net.linear2, True)

    def test_basic_layer_gradient_shap(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(
            net, LayerGradientShap, net.linear1,
        )

    def test_basic_layer_gradient_shap_with_alt_devices(self):
        net = BasicModel_MultiLayer(inplace=True).cuda()
        self._basic_gradient_shap_helper(
            net, LayerGradientShap, net.linear1, True,
        )

    def _basic_gradient_shap_helper(
        self, net, attr_method_class, layer, alt_device_ids=False
    ):
        net.eval()
        inputs = torch.tensor([[1.0, -20.0, 10.0], [11.0, 10.0, -11.0]]).cuda()
        baselines = torch.randn(30, 3).cuda()
        if attr_method_class == NeuronGradientShap:
            self._data_parallel_test_assert(
                attr_method_class,
                net,
                layer,
                alt_device_ids=alt_device_ids,
                inputs=inputs,
                neuron_index=0,
                baselines=baselines,
                additional_forward_args=None,
                test_batches=False,
            )
        else:
            self._data_parallel_test_assert(
                attr_method_class,
                net,
                layer,
                alt_device_ids=alt_device_ids,
                inputs=inputs,
                target=0,
                baselines=baselines,
                additional_forward_args=None,
                test_batches=False,
            )

    def test_multi_input_neuron_ablation(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()
        for perturbations_per_eval in [1, 2, 3]:
            self._data_parallel_test_assert(
                NeuronFeatureAblation,
                net,
                net.l3,
                inputs=(inp1, inp2),
                neuron_index=0,
                additional_forward_args=None,
                test_batches=False,
                perturbations_per_eval=perturbations_per_eval,
                alt_device_ids=True,
            )

    def test_multi_input_neuron_ablation_with_baseline(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True).cuda()

        base1 = torch.tensor([[1.0, 0.0, 1.0]], requires_grad=True).cuda()
        base2 = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True).cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                NeuronFeatureAblation,
                net,
                net.l3,
                inputs=(inp1, inp2),
                neuron_index=0,
                baselines=(base1, base2),
                additional_forward_args=None,
                test_batches=False,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_simple_feature_ablation(self):
        net = BasicModel_ConvNet().cuda()
        inp = torch.arange(400).view(4, 1, 10, 10).type(torch.FloatTensor).cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                FeatureAblation,
                net,
                None,
                inputs=inp,
                target=0,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_simple_occlusion(self):
        net = BasicModel_ConvNet().cuda()
        inp = torch.arange(400).view(4, 1, 10, 10).float().cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                Occlusion,
                net,
                None,
                inputs=inp,
                sliding_window_shapes=(1, 4, 2),
                target=0,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_multi_input_occlusion(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]]).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]]).cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                Occlusion,
                net,
                None,
                inputs=(inp1, inp2),
                sliding_window_shapes=((2,), (1,)),
                test_batches=False,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_simple_shapley_sampling(self):
        net = BasicModel_ConvNet().cuda()
        inp = torch.arange(400).view(4, 1, 10, 10).float().cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                ShapleyValueSampling,
                net,
                None,
                inputs=inp,
                target=0,
                perturbations_per_eval=perturbations_per_eval,
            )

    def test_multi_input_shapley_sampling(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]]).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]]).cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                ShapleyValueSampling,
                net,
                None,
                inputs=(inp1, inp2),
                test_batches=False,
                perturbations_per_eval=perturbations_per_eval,
                delta=0.1,
            )

    def test_multi_input_shapley_values(self):
        net = ReLULinearDeepLiftModel().cuda()
        inp1 = torch.tensor([[-10.0, 1.0, -5.0]]).cuda()
        inp2 = torch.tensor([[3.0, 3.0, 1.0]]).cuda()
        for perturbations_per_eval in [1, 8, 20]:
            self._data_parallel_test_assert(
                ShapleyValues,
                net,
                None,
                inputs=(inp1, inp2),
                test_batches=False,
                perturbations_per_eval=perturbations_per_eval,
            )

    def _alt_device_list(self):
        return [0] + [x for x in range(torch.cuda.device_count() - 1, 0, -1)]

    def _data_parallel_test_assert(
        self,
        algorithm,
        model,
        target_layer=None,
        alt_device_ids=False,
        test_batches=False,
        delta=0.0001,
        **kwargs
    ):
        if alt_device_ids:
            dp_model = torch.nn.parallel.DataParallel(
                model, device_ids=self._alt_device_list()
            )
        else:
            dp_model = torch.nn.parallel.DataParallel(model)
        if target_layer:
            attr_orig = algorithm(model, target_layer)
            if alt_device_ids:
                attr_dp = algorithm(
                    dp_model.forward, target_layer, device_ids=self._alt_device_list()
                )
            else:
                attr_dp = algorithm(dp_model, target_layer)
        else:
            attr_orig = algorithm(model)
            attr_dp = algorithm(dp_model)

        batch_sizes = [None]
        delta_orig = None
        delta_dp = None
        if test_batches:
            batch_sizes = [None, 1, 8]
        for batch_size in batch_sizes:
            if batch_size:
                attributions_orig = attr_orig.attribute(
                    internal_batch_size=batch_size, **kwargs
                )
            else:
                if attr_orig.has_convergence_delta():
                    attributions_orig, delta_orig = attr_orig.attribute(
                        return_convergence_delta=True, **kwargs
                    )
                else:
                    attributions_orig = attr_orig.attribute(**kwargs)
            self.setUp()
            if batch_size:
                attributions_dp = attr_dp.attribute(
                    internal_batch_size=batch_size, **kwargs
                )
            else:
                if attr_orig.has_convergence_delta():
                    attributions_dp, delta_dp = attr_dp.attribute(
                        return_convergence_delta=True, **kwargs
                    )
                else:
                    attributions_dp = attr_dp.attribute(**kwargs)
            if isinstance(attributions_dp, torch.Tensor):
                self.assertAlmostEqual(
                    torch.sum(torch.abs(attributions_orig - attributions_dp)),
                    0,
                    delta=delta,
                )
            else:
                for i in range(len(attributions_orig)):
                    self.assertAlmostEqual(
                        torch.sum(torch.abs(attributions_orig[i] - attributions_dp[i])),
                        0,
                        delta=delta,
                    )

            if delta_dp is not None:
                self.assertAlmostEqual(
                    torch.sum(torch.abs(delta_orig - delta_dp)), 0, delta=delta
                )


if __name__ == "__main__":
    unittest.main()
