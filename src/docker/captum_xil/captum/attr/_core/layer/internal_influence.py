#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from ..._utils.approximation_methods import approximation_parameters
from ..._utils.attribution import GradientAttribution, LayerAttribution
from ..._utils.batching import _batched_operator
from ..._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_attributions,
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from ..._utils.gradient import compute_layer_gradients_and_eval
from ..._utils.typing import BaselineType, TargetType


class InternalInfluence(LayerAttribution, GradientAttribution):
    r"""
    Computes internal influence by approximating the integral of gradients
    for a particular layer along the path from a baseline input to the
    given input.
    If no baseline is provided, the default baseline is the zero tensor.
    More details on this approach can be found here:
    https://arxiv.org/pdf/1802.03788.pdf

    Note that this method is similar to applying integrated gradients and
    taking the layer as input, integrating the gradient of the layer with
    respect to the output.
    """

    def __init__(
        self,
        forward_func: Callable,
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        attribute_to_layer_input: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which internal
                        influence is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define a starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
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
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (string, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations
                        are processed in one batch.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer inputs, otherwise it will be computed with respect
                        to layer outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Internal influence of each neuron in given
                        layer output. Attributions will always be the same size
                        as the output or input of the given layer depending on
                        whether `attribute_to_layer_input` is set to `False` or
                        `True`respectively.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> layer_int_inf = InternalInfluence(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer internal influence.
            >>> # attribution size matches layer output, Nx12x32x32
            >>> attribution = layer_int_inf.attribute(input)
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines, n_steps, method)

        # Retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # Compute scaled inputs from baseline to final input.
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # Returns gradient of output with respect to hidden layer.
        layer_gradients, _, is_layer_tuple = _batched_operator(
            compute_layer_gradients_and_eval,
            scaled_features_tpl,
            input_additional_args,
            internal_batch_size=internal_batch_size,
            forward_fn=self.forward_func,
            layer=self.layer,
            target_ind=expanded_target,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        # flattening grads so that we can multiply it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = tuple(
            layer_grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(layer_grad.device)
            for layer_grad in layer_gradients
        )

        # aggregates across all steps for each tensor in the input tuple
        attrs = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, inputs[0].shape[0], layer_grad.shape[1:]
            )
            for scaled_grad, layer_grad in zip(scaled_grads, layer_gradients)
        )
        return _format_attributions(is_layer_tuple, attrs)
