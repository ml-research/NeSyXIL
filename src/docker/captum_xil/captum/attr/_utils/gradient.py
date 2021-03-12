#!/usr/bin/env python3
import threading
import typing
import warnings
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import torch
from torch import Tensor, device
from torch.nn import Module

from .batching import _reduce_list, _sort_key_list
from .common import _run_forward, _verify_select_column
from .typing import Literal, TargetType, TensorOrTupleOfTensorsGeneric


def apply_gradient_requirements(inputs: Tuple[Tensor, ...]) -> List[bool]:
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.
    """
    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients"
    grad_required = []
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(input.requires_grad)
        if not input.requires_grad:
            warnings.warn(
                "Input Tensor %d did not already require gradients, "
                "required_grads has been set automatically." % index
            )
            input.requires_grad_()
        if input.grad is not None:
            if torch.sum(torch.abs(input.grad)).item() > 1e-7:
                warnings.warn(
                    "Input Tensor %d had a non-zero gradient tensor, "
                    "which is being reset to 0." % index
                )
            input.grad.zero_()
    return grad_required


def undo_gradient_requirements(
    inputs: Tuple[Tensor, ...], grad_required: List[bool]
) -> None:
    """
    Iterates through list of tensors, zeros each gradient, and sets required
    grad to false if the corresponding index in grad_required is False.
    This method is used to undo the effects of prepare_gradient_inputs, making
    grads not required for any input tensor that did not initially require
    gradients.
    """

    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients."
    assert len(inputs) == len(
        grad_required
    ), "Input tuple length should match gradient mask."
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        if input.grad is not None:
            input.grad.detach_()
            input.grad.zero_()
        if not grad_required[index]:
            input.requires_grad_(False)


def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
        Computes gradients of the output with respect to inputs for an
        arbitrary forward function.

        Args:

            forward_fn: forward function. This can be for example model's
                        forward function.
            input:      Input at which gradients are evaluated,
                        will be passed to forward_fn.
            target_ind: Index of the target class for which gradients
                        must be computed (classification only).
            additional_forward_args: Additional input arguments that forward
                        function requires. It takes an empty tuple (no additional
                        arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # This line updated by Wolfgang Stammer to retrieve the gradients of the explanations
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        #grads = torch.autograd.grad(torch.unbind(outputs), inputs)
        grads = torch.autograd.grad(torch.unbind(outputs), inputs, create_graph=True)
    return grads


def _neuron_gradients(
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    saved_layer: Dict[device, Tuple[Tensor, ...]],
    key_list: List[device],
    gradient_neuron_index: Union[int, Tuple[int, ...]],
) -> Tuple[Tensor, ...]:
    with torch.autograd.set_grad_enabled(True):
        gradient_tensors = []
        for key in key_list:
            assert (
                len(saved_layer[key]) == 1
            ), "Cannot compute neuron gradients for layer with multiple tensors."
            current_out_tensor = saved_layer[key][0]
            gradient_tensors.append(
                torch.autograd.grad(
                    torch.unbind(
                        _verify_select_column(current_out_tensor, gradient_neuron_index)
                    ),
                    inputs,
                )
            )
        _total_gradients = _reduce_list(gradient_tensors, sum)
    return _total_gradients


def _forward_layer_eval(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Any = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[Tuple[Tensor, ...], Literal[True, False]]:
    return _forward_layer_eval_with_neuron_grads(
        forward_fn,
        inputs,
        layer,
        additional_forward_args=additional_forward_args,
        gradient_neuron_index=None,
        device_ids=device_ids,
        attribute_to_layer_input=attribute_to_layer_input,
    )


@typing.overload
def _forward_layer_distributed_eval(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    attribute_to_layer_input: bool = False,
    forward_hook_with_return: Literal[False] = False,
) -> Tuple[Dict[device, Tuple[Tensor, ...]], Literal[True, False]]:
    ...


@typing.overload
def _forward_layer_distributed_eval(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    attribute_to_layer_input: bool = False,
    *,
    forward_hook_with_return: Literal[True],
) -> Tuple[Dict[device, Tuple[Tensor, ...]], Tensor, Literal[True, False]]:
    ...


def _forward_layer_distributed_eval(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    attribute_to_layer_input: bool = False,
    forward_hook_with_return: bool = False,
) -> Union[
    Tuple[Dict[device, Tuple[Tensor, ...]], Tensor, bool],
    Tuple[Dict[device, Tuple[Tensor, ...]], bool],
]:
    r"""
    A helper function that allows to set a hook on model's `layer`, run the forward
    pass and returns intermediate layer results, stored in a dictionary,
    and optionally also the output of the forward function. The keys in the
    dictionary are the device ids and the values are corresponding intermediate layer
    results, either the inputs or the outputs of the layer depending on whether we set
    `attribute_to_layer_input` to True or False.
    This is especially useful when we execute forward pass in a distributed setting,
    using `DataParallel`s for example.
    """
    saved_layer = {}
    is_eval_tuple = False
    lock = threading.Lock()
    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor(s).
    # For DataParallel models, each partition adds entry to dictionary
    # with key as device and value as corresponding Tensor.

    def forward_hook(module, inp, out=None):
        nonlocal is_eval_tuple
        eval_tsrs = inp if attribute_to_layer_input else out
        is_eval_tuple = isinstance(eval_tsrs, tuple)
        if not is_eval_tuple:
            eval_tsrs = (eval_tsrs,)
        with lock:
            nonlocal saved_layer
            # Note that cloning behaviour of `eval_tsr` is different
            # when `forward_hook_with_return` is set to True. This is because
            # otherwise `backward()` on the last output layer won't execute.
            if forward_hook_with_return:
                saved_layer[eval_tsrs[0].device] = eval_tsrs
                eval_tsrs_to_return = tuple(eval_tsr.clone() for eval_tsr in eval_tsrs)
                if not is_eval_tuple:
                    eval_tsrs_to_return = eval_tsrs_to_return[0]
                return eval_tsrs_to_return
            else:
                saved_layer[eval_tsrs[0].device] = tuple(
                    eval_tsr.clone() for eval_tsr in eval_tsrs
                )

    if attribute_to_layer_input:
        hook = layer.register_forward_pre_hook(forward_hook)
    else:
        hook = layer.register_forward_hook(forward_hook)
    output = _run_forward(
        forward_fn,
        inputs,
        target=target_ind,
        additional_forward_args=additional_forward_args,
    )
    hook.remove()

    if len(saved_layer) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    if forward_hook_with_return:
        return saved_layer, output, is_eval_tuple
    return saved_layer, is_eval_tuple


def _gather_distributed_tensors(
    saved_layer: Dict[device, Tuple[Tensor, ...]],
    device_ids: Union[None, List[int]] = None,
    key_list: Union[None, List[device]] = None,
) -> Tuple[Tensor, ...]:
    r"""
    A helper function to concatenate intermediate layer results stored on
    different devices in `saved_layer`. `saved_layer` is a dictionary that
    contains `device_id` as a key and intermediate layer results (either
    the input or the output of the layer) stored on the device corresponding to
    the key.
    `key_list` is a list of devices in appropriate ordering for concatenation
    and if not provided, keys are sorted based on device ids.

    If only one key exists (standard model), key list simply has one element.
    """
    if key_list is None:
        key_list = _sort_key_list(list(saved_layer.keys()), device_ids)
    return _reduce_list([saved_layer[device_id] for device_id in key_list])


def _extract_device_ids(
    forward_fn: Callable,
    saved_layer: Dict[device, Tuple[Tensor, ...]],
    device_ids: Union[None, List[int]],
) -> Union[None, List[int]]:
    r"""
    A helper function to extract device_ids from `forward_function` in case it is
    provided as part of a `DataParallel` model or if is accessible from
    `forward_fn`.
    In case input device_ids is not None, this function returns that value.
    """
    # Multiple devices / keys implies a DataParallel model, so we look for
    # device IDs if given or available from forward function
    # (DataParallel model object).
    if len(saved_layer) > 1 and device_ids is None:
        if (
            hasattr(forward_fn, "device_ids")
            and cast(Any, forward_fn).device_ids is not None
        ):
            device_ids = cast(Any, forward_fn).device_ids
        else:
            raise AssertionError(
                "Layer tensors are saved on multiple devices, however unable to access"
                " device ID list from the `forward_fn`. Device ID list must be"
                " accessible from `forward_fn`. For example, they can be retrieved"
                " if `forward_fn` is a model of type `DataParallel`. It is used"
                " for identifying device batch ordering."
            )
    return device_ids


@typing.overload
def _forward_layer_eval_with_neuron_grads(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Any = None,
    *,
    gradient_neuron_index: Union[int, Tuple[int, ...]],
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Literal[True, False]]:
    ...


@typing.overload
def _forward_layer_eval_with_neuron_grads(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Any = None,
    gradient_neuron_index: None = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[Tuple[Tensor, ...], Literal[True, False]]:
    ...


def _forward_layer_eval_with_neuron_grads(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Any = None,
    gradient_neuron_index: Union[None, int, Tuple[int, ...]] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], bool], Tuple[Tuple[Tensor, ...], bool]
]:
    """
    This method computes forward evaluation for a particular layer using a
    forward hook. If a gradient_neuron_index is provided, then gradients with
    respect to that neuron in the layer output are also returned.

    These functionalities are combined due to the behavior of DataParallel models
    with hooks, in which hooks are executed once per device. We need to internally
    combine the separated tensors from devices by concatenating based on device_ids.
    Any necessary gradients must be taken with respect to each independent batched
    tensor, so the gradients are computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel models
    can be found in the PyTorch data parallel documentation. We maintain the separate
    evals in a dictionary protected by a lock, analogous to the gather implementation
    for the core PyTorch DataParallel implementation.
    """
    saved_layer, is_layer_tuple = _forward_layer_distributed_eval(
        forward_fn,
        inputs,
        layer,
        additional_forward_args=additional_forward_args,
        attribute_to_layer_input=attribute_to_layer_input,
    )
    device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)
    # Identifies correct device ordering based on device ids.
    # key_list is a list of devices in appropriate ordering for concatenation.
    # If only one key exists (standard model), key list simply has one element.
    key_list = _sort_key_list(list(saved_layer.keys()), device_ids)
    if gradient_neuron_index is not None:
        inp_grads = _neuron_gradients(
            inputs, saved_layer, key_list, gradient_neuron_index
        )
        return (
            _gather_distributed_tensors(saved_layer, key_list=key_list),
            inp_grads,
            is_layer_tuple,
        )
    else:
        return (
            _gather_distributed_tensors(saved_layer, key_list=key_list),
            is_layer_tuple,
        )


@typing.overload
def compute_layer_gradients_and_eval(
    forward_fn: Callable,
    layer: Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    *,
    gradient_neuron_index: Union[int, Tuple[int, ...]],
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    output_fn: Union[None, Callable] = None,
) -> Tuple[
    Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...], Literal[True, False],
]:
    ...


@typing.overload
def compute_layer_gradients_and_eval(
    forward_fn: Callable,
    layer: Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    gradient_neuron_index: None = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    output_fn: Union[None, Callable] = None,
) -> Tuple[
    Tuple[Tensor, ...], Tuple[Tensor, ...], Literal[True, False],
]:
    ...


def compute_layer_gradients_and_eval(
    forward_fn: Callable,
    layer: Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    gradient_neuron_index: Union[None, int, Tuple[int, ...]] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    output_fn: Union[None, Callable] = None,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], bool],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...], bool],
]:
    r"""
        Computes gradients of the output with respect to a given layer as well
        as the output evaluation of the layer for an arbitrary forward function
        and given input.

        For data parallel models, hooks are executed once per device ,so we
        need to internally combine the separated tensors from devices by
        concatenating based on device_ids. Any necessary gradients must be taken
        with respect to each independent batched tensor, so the gradients are
        computed and combined appropriately.

        More information regarding the behavior of forward hooks with DataParallel
        models can be found in the PyTorch data parallel documentation. We maintain
        the separate inputs in a dictionary protected by a lock, analogous to the
        gather implementation for the core PyTorch DataParallel implementation.

        NOTE: To properly handle inplace operations, a clone of the layer output
        is stored. This structure inhibits execution of a backward hook on the last
        module for the layer output when computing the gradient with respect to
        the input, since we store an intermediate clone, as
        opposed to the true module output. If backward module hooks are necessary
        for the final module when computing input gradients, utilize
        _forward_layer_eval_with_neuron_grads instead.

        Args:

            forward_fn: forward function. This can be for example model's
                        forward function.
            layer:      Layer for which gradients / output will be evaluated.
            inputs:     Input at which gradients are evaluated,
                        will be passed to forward_fn.
            target_ind: Index of the target class for which gradients
                        must be computed (classification only).
            output_fn:  An optional function that is applied to the layer inputs or
                        outputs depending whether the `attribute_to_layer_input` is
                        set to `True` or `False`
            args:       Additional input arguments that forward function requires.
                        It takes an empty tuple (no additional arguments) if no
                        additional arguments are required


        Returns:
            2-element tuple of **gradients**, **evals**:
            - **gradients**:
                Gradients of output with respect to target layer output.
            - **evals**:
                Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        # saved_layer is a dictionary mapping device to a tuple of
        # layer evaluations on that device.
        saved_layer, output, is_layer_tuple = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            forward_hook_with_return=True,
        )
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(list(saved_layer.keys()), device_ids)

        all_outputs = _reduce_list(
            [
                saved_layer[device_id]
                if output_fn is None
                else output_fn(saved_layer[device_id])
                for device_id in key_list
            ]
        )
        num_tensors = len(saved_layer[next(iter(saved_layer))])
        grad_inputs = tuple(
            layer_tensor
            for device_id in key_list
            for layer_tensor in saved_layer[device_id]
        )
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs)
        saved_grads = [
            saved_grads[i : i + num_tensors]
            for i in range(0, len(saved_grads), num_tensors)
        ]
        if output_fn is not None:
            saved_grads = [output_fn(saved_grad) for saved_grad in saved_grads]

        all_grads = _reduce_list(saved_grads)
        if gradient_neuron_index is not None:
            inp_grads = _neuron_gradients(
                inputs, saved_layer, key_list, gradient_neuron_index
            )
            return all_grads, all_outputs, inp_grads, is_layer_tuple
    return all_grads, all_outputs, is_layer_tuple


def construct_neuron_grad_fn(
    layer: Module,
    neuron_index: Union[int, Tuple[int, ...]],
    device_ids: Union[None, List[int]] = None,
    attribute_to_neuron_input: bool = False,
) -> Callable:
    def grad_fn(
        forward_fn: Callable,
        inputs: TensorOrTupleOfTensorsGeneric,
        target_ind: TargetType = None,
        additional_forward_args: Any = None,
    ) -> Tuple[Tensor, ...]:
        _, grads, _ = _forward_layer_eval_with_neuron_grads(
            forward_fn,
            inputs,
            layer,
            additional_forward_args,
            gradient_neuron_index=neuron_index,
            device_ids=device_ids,
            attribute_to_layer_input=attribute_to_neuron_input,
        )
        return grads

    return grad_fn
