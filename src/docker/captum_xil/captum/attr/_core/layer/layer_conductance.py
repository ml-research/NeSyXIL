#!/usr/bin/env python3
import typing
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
from ..._utils.typing import BaselineType, Literal, TargetType


class LayerConductance(LayerAttribution, GradientAttribution):
    r"""
    Computes conductance with respect to the given layer. The
    returned output is in the shape of the layer's output, showing the total
    conductance of each hidden layer neuron.

    The details of the approach can be found here:
    https://arxiv.org/abs/1805.12233
    https://arxiv.org/pdf/1807.09946.pdf

    Note that this provides the total conductance of each neuron in the
    layer's output. To obtain the breakdown of a neuron's conductance by input
    features, utilize NeuronConductance instead, and provide the target
    neuron index.
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

    def has_convergence_delta(self) -> bool:
        return True

    @typing.overload
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool = False,
    ) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        ...

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Union[
            None, int, float, Tensor, Tuple[Union[int, float, Tensor], ...]
        ] = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[
        Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor],
    ]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which layer
                        conductance is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define the starting point from which integral
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
                        correspond to the number of examples. It will be repeated
                        for each of `n_steps` along the integrated path.
                        For all other types, the given argument is used for
                        all forward evaluations.
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
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
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
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Conductance of each neuron in given layer input or
                        output. Attributions will always be the same size as
                        the input or output of the given layer, depending on
                        whether we attribute to the inputs or outputs
                        of the layer which is decided by the input flag
                        `attribute_to_layer_input`.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                        The difference between the total
                        approximated and true conductance.
                        This is computed using the property that the total sum of
                        forward_func(inputs) - forward_func(baselines) must equal
                        the total sum of the attributions.
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in inputs.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> layer_cond = LayerConductance(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer conductance for class 3.
            >>> # attribution size matches layer output, Nx12x32x32
            >>> attribution = layer_cond.attribute(input, target=3)
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines, n_steps, method)

        num_examples = inputs[0].shape[0]

        # Retrieve scaling factors for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        alphas = alphas_func(n_steps + 1)

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
        # dim -> (#examples * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps + 1)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps + 1)

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals, is_layer_tuple = _batched_operator(
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

        # Compute differences between consecutive evaluations of layer_eval.
        # This approximates the total input gradient of each step multiplied
        # by the step size.
        grad_diffs = tuple(
            layer_eval[num_examples:] - layer_eval[:-num_examples]
            for layer_eval in layer_evals
        )

        # Element-wise multiply gradient of output with respect to hidden layer
        # and summed gradients with respect to input (chain rule) and sum
        # across stepped inputs.
        attributions = tuple(
            _reshape_and_sum(
                grad_diff * layer_gradient[:-num_examples],
                n_steps,
                num_examples,
                layer_eval.shape[1:],
            )
            for layer_gradient, layer_eval, grad_diff in zip(
                layer_gradients, layer_evals, grad_diffs
            )
        )

        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                target=target,
                additional_forward_args=additional_forward_args,
            )
            return _format_attributions(is_layer_tuple, attributions), delta
        return _format_attributions(is_layer_tuple, attributions)
