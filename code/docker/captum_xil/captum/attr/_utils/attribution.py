#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Type, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .common import (
    _format_additional_forward_args,
    _format_input_baseline,
    _format_tensor_into_tuples,
    _run_forward,
    _tensorize_baseline,
    _validate_input,
    _validate_target,
)
from .gradient import compute_gradients
from .typing import TargetType


class Attribution:
    r"""
    All attribution algorithms extend this class. It enforces its child classes
    to extend and override core `attribute` method.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:
            forward_func (callable or torch.nn.Module): This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
        """
        self.forward_func = forward_func

    attribute: Callable
    r"""
    This method computes and returns the attribution values for each input tensor.
    Deriving classes are responsible for implementing its logic accordingly.

    Specific attribution algorithms that extend this class take relevant
    arguments.

    Args:

        inputs (tensor or tuple of tensors):  Input for which attribution
                    is computed. It can be provided as a single tensor or
                    a tuple of multiple tensors. If multiple input tensors
                    are provided, the batch sizes must be aligned accross all
                    tensors.


    Returns:

        *tensor* or tuple of *tensors* of **attributions**:
        - **attributions** (*tensor* or tuple of *tensors*):
                    Attribution values for each
                    input tensor. The `attributions` have the same shape and
                    dimensionality as the inputs.
                    If a single tensor is provided as inputs, a single tensor
                    is returned. If a tuple is provided for inputs, a tuple of
                    corresponding sized tensors is returned.

    """

    def has_convergence_delta(self) -> bool:
        r"""
        This method informs the user whether the attribution algorithm provides
        a convergence delta (aka an approximation error) or not. Convergence
        delta may serve as a proxy of correctness of attribution algorithm's
        approximation. If deriving attribution class provides a
        `compute_convergence_delta` method, it should
        override both `compute_convergence_delta` and `has_convergence_delta` methods.

        Returns:
            bool:
            Returns whether the attribution algorithm
            provides a convergence delta (aka approximation error) or not.

        """
        return False

    compute_convergence_delta: Callable
    r"""
    The attribution algorithms which derive `Attribution` class and provide
    convergence delta (aka approximation error) should implement this method.
    Convergence delta can be computed based on certain properties of the
    attribution alogrithms.

    Args:

            attributions (tensor or tuple of tensors): Attribution scores that
                        are precomputed by an attribution algorithm.
                        Attributions can be provided in form of a single tensor
                        or a tuple of those. It is assumed that attribution
                        tensor's dimension 0 corresponds to the number of
                        examples, and if multiple input tensors are provided,
                        the examples must be aligned appropriately.
            *args (optional): Additonal arguments that are used by the
                        sub-classes depending on the specific implementation
                        of `compute_convergence_delta`.

    Returns:

            *tensor* of **deltas**:
            - **deltas** (*tensor*):
                Depending on specific implementaion of
                sub-classes, convergence delta can be returned per
                sample in form of a tensor or it can be aggregated
                across multuple samples and returned in form of a
                single floating point tensor.
    """

    @classmethod
    def get_name(cls: Type["Attribution"]) -> str:
        r"""
        Create readable class name by inserting a space before any capital
        characters besides the very first.

        Returns:
            str: a readable class name
        Example:
            for a class called IntegratedGradients, we return the string
            'Integrated Gradients'
        """
        return "".join(
            [
                char if char.islower() or idx == 0 else " " + char
                for idx, char in enumerate(cls.__name__)
            ]
        )


class GradientAttribution(Attribution):
    r"""
    All gradient based attribution algorithms extend this class. It requires a
    forward function, which most commonly is the forward function of the model
    that we want to interpret or the model itself.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable or torch.nn.Module): This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
        """
        Attribution.__init__(self, forward_func)
        self.gradient_func = compute_gradients

    def compute_convergence_delta(
        self,
        attributions: Union[Tensor, Tuple[Tensor, ...]],
        start_point: Union[
            None, int, float, Tensor, Tuple[Union[int, float, Tensor], ...]
        ],
        end_point: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
    ) -> Tensor:
        r"""
        Here we provide a specific implementation for `compute_convergence_delta`
        which is based on a common property among gradient-based attribution algorithms.
        In the literature sometimes it is also called completeness axiom. Completeness
        axiom states that the sum of the attribution must be equal to the differences of
        NN Models's function at its end and start points. In other words:
        sum(attributions) - (F(end_point) - F(start_point)) is close to zero.
        Returned delta of this method is defined as above stated difference.

        This implementation assumes that both the `start_point` and `end_point` have
        the same shape and dimensionality. It also assumes that the target must have
        the same number of examples as the `start_point` and the `end_point` in case
        it is provided in form of a list or a non-singleton tensor.

        Args:

                attributions (tensor or tuple of tensors): Precomputed attribution
                            scores. The user can compute those using any attribution
                            algorithm. It is assumed the the shape and the
                            dimensionality of attributions must match the shape and
                            the dimensionality of `start_point` and `end_point`.
                            It also assumes that the attribution tensor's
                            dimension 0 corresponds to the number of
                            examples, and if multiple input tensors are provided,
                            the examples must be aligned appropriately.
                start_point (tensor or tuple of tensors, optional): `start_point`
                            is passed as an input to model's forward function. It
                            is the starting point of attributions' approximation.
                            It is assumed that both `start_point` and `end_point`
                            have the same shape and dimensionality.
                end_point (tensor or tuple of tensors):  `end_point`
                            is passed as an input to model's forward function. It
                            is the end point of attributions' approximation.
                            It is assumed that both `start_point` and `end_point`
                            have the same shape and dimensionality.
                target (int, tuple, tensor or list, optional):  Output indices for
                            which gradients are computed (for classification cases,
                            this is usually the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary.
                            For general 2D outputs, targets can be either:

                            - a single integer or a tensor containing a single
                              integer, which is applied to all input examples

                            - a list of integers or a 1D tensor, with length matching
                              the number of examples in inputs (dim 0). Each integer
                              is applied as the target for the corresponding example.

                            For outputs with > 2 dimensions, targets can be either:

                            - A single tuple, which contains #output_dims - 1
                              elements. This target index is applied to all examples.

                            - A list of tuples with length equal to the number of
                              examples in inputs (dim 0), and each tuple containing
                              #output_dims - 1 elements. Each tuple is applied as the
                              target for the corresponding example.

                            Default: None
                additional_forward_args (any, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be either a single additional
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            For a tensor, the first dimension of the tensor must
                            correspond to the number of examples.
                            `additional_forward_args` is used both for `start_point`
                            and `end_point` when computing the forward pass.
                            Default: None

        Returns:

                *tensor* of **deltas**:
                - **deltas** (*tensor*):
                    This implementation returns convergence delta per
                    sample. Deriving sub-classes may do any type of aggregation
                    of those values, if necessary.
        """
        end_point, start_point = _format_input_baseline(end_point, start_point)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # tensorizing start_point in case it is a scalar or one example baseline
        # If the batch size is large we could potentially also tensorize only one
        # sample and expand the output to the rest of the elements in the batch
        start_point = _tensorize_baseline(end_point, start_point)

        attributions = _format_tensor_into_tuples(attributions)

        # verify that the attributions and end_point match on 1st dimension
        for attribution, end_point_tnsr in zip(attributions, end_point):
            assert end_point_tnsr.shape[0] == attribution.shape[0], (
                "Attributions tensor and the end_point must match on the first"
                " dimension but found attribution: {} and end_point: {}".format(
                    attribution.shape[0], end_point_tnsr.shape[0]
                )
            )

        num_samples = end_point[0].shape[0]
        _validate_input(end_point, start_point)
        _validate_target(num_samples, target)

        def _sum_rows(input: Tensor) -> Tensor:
            return input.view(input.shape[0], -1).sum(1)

        with torch.no_grad():
            start_out_sum = _sum_rows(
                _run_forward(
                    self.forward_func, start_point, target, additional_forward_args
                )
            )

            end_out_sum = _sum_rows(
                _run_forward(
                    self.forward_func, end_point, target, additional_forward_args
                )
            )
            row_sums = [_sum_rows(attribution) for attribution in attributions]
            attr_sum = torch.stack(
                [cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)]
            )
            _delta = attr_sum - (end_out_sum - start_out_sum)
        return _delta


class PerturbationAttribution(Attribution):
    r"""
    All perturbation based attribution algorithms extend this class. It requires a
    forward function, which most commonly is the forward function of the model
    that we want to interpret or the model itself.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable or torch.nn.Module): This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
        """
        Attribution.__init__(self, forward_func)


class InternalAttribution(Attribution):
    r"""
    Shared base class for LayerAttrubution and NeuronAttribution,
    attribution types that require a model and a particular layer.
    """

    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (callable or torch.nn.Module):  This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
            layer (torch.nn.Module): Layer for which output attributions are computed.
                        Output size of attribute matches that of layer output.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                        applies a DataParallel model, which allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not necessary to provide this argument.
        """
        Attribution.__init__(self, forward_func)
        self.layer = layer
        self.device_ids = device_ids


class LayerAttribution(InternalAttribution):
    r"""
    Layer attribution provides attribution values for the given layer, quanitfying
    the importance of each neuron within the given layer's output. The output
    attribution of calling attribute on a LayerAttribution object always matches
    the size of the layer output.
    """

    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (callable or torch.nn.Module):  This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
            layer (torch.nn.Module): Layer for which output attributions are computed.
                        Output size of attribute matches that of layer output.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                        applies a DataParallel model, which allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not necessary to provide this argument.
        """
        InternalAttribution.__init__(self, forward_func, layer, device_ids)

    @staticmethod
    def interpolate(
        layer_attribution: Tensor,
        interpolate_dims: Union[int, Tuple[int, ...]],
        interpolate_mode: str = "nearest",
    ) -> Tensor:
        r"""
        Interpolates given 3D, 4D or 5D layer attribution to given dimensions.
        This is often utilized to upsample the attribution of a convolutional layer
        to the size of an input, which allows visualizing in the input space.

        Args:

            layer_attribution (torch.Tensor):  Tensor of given layer attributions.
            interpolate_dims (int or tuple): Upsampled dimensions. The
                        number of elements must be the number of dimensions
                        of layer_attribution - 2, since the first dimension
                        corresponds to number of examples and the second is
                        assumed to correspond to the number of channels.
            interpolate_mode (str):  Method for interpolation, which
                        must be a valid input interpolation mode for
                        torch.nn.functional. These methods are
                        "nearest", "area", "linear" (3D-only), "bilinear"
                        (4D-only), "bicubic" (4D-only), "trilinear" (5D-only)
                        based on the number of dimensions of the given layer
                        attribution.

        Returns:
            *tensor* of upsampled **attributions**:
            - **attributions** (*tensor*):
                Upsampled layer attributions with first 2 dimensions matching
                slayer_attribution and remaining dimensions given by
                interpolate_dims.
        """
        return F.interpolate(layer_attribution, interpolate_dims, mode=interpolate_mode)


class NeuronAttribution(InternalAttribution):
    r"""
    Neuron attribution provides input attribution for a given neuron, quanitfying
    the importance of each input feature in the activation of a particular neuron.
    Calling attribute on a NeuronAttribution object requires also providing
    the index of the neuron in the output of the given layer for which attributions
    are required.
    The output attribution of calling attribute on a NeuronAttribution object
    always matches the size of the input.
    """

    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (callable or torch.nn.Module):  This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
            layer (torch.nn.Module): Layer for which output attributions are computed.
                        Output size of attribute matches that of layer output.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                        applies a DataParallel model, which allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not necessary to provide this argument.
        """
        InternalAttribution.__init__(self, forward_func, layer, device_ids)

    attribute: Callable
    r"""
    This method computes and returns the neuron attribution values for each
    input tensor. Deriving classes are responsible for implementing
    its logic accordingly.

    Specific attribution algorithms that extend this class take relevant
    arguments.

    Args:

            inputs:     A single high dimensional input tensor or a tuple of them.
            neuron_index (int or tuple): Tuple providing index of neuron in output
                    of given layer for which attribution is desired. Length of
                    this tuple must be one less than the number of
                    dimensions in the output of the given layer (since
                    dimension 0 corresponds to number of examples).

    Returns:

            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                    Attribution values for
                    each input vector. The `attributions` have the
                    dimensionality of inputs.
    """
