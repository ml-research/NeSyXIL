#!/usr/bin/env python3
import typing
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import torch
from torch import Tensor, device

from .common import _format_additional_forward_args, _format_input
from .typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
    TupleOrTensorOrBoolGeneric,
)


@typing.overload
def _tuple_splice_range(inputs: None, start: int, end: int) -> None:
    ...


@typing.overload
def _tuple_splice_range(inputs: Tuple, start: int, end: int) -> Tuple:
    ...


def _tuple_splice_range(
    inputs: Union[None, Tuple], start: int, end: int
) -> Union[None, Tuple]:
    """
    Splices each tensor element of given tuple (inputs) from range start
    (inclusive) to end (non-inclusive) on its first dimension. If element
    is not a Tensor, it is left unchanged. It is assumed that all tensor elements
    have the same first dimension (corresponding to number of examples).
    The returned value is a tuple with the same length as inputs, with Tensors
    spliced appropriately.
    """
    assert start < end, "Start point must precede end point for batch splicing."
    if inputs is None:
        return None
    return tuple(
        inp[start:end] if isinstance(inp, torch.Tensor) else inp for inp in inputs
    )


def _reduce_list(
    val_list: List[TupleOrTensorOrBoolGeneric],
    red_func: Callable[[List], Any] = torch.cat,
) -> TupleOrTensorOrBoolGeneric:
    """
    Applies reduction function to given list. If each element in the list is
    a Tensor, applies reduction function to all elements of the list, and returns
    the output Tensor / value. If each element is a boolean, apply any method (or).
    If each element is a tuple, applies reduction
    function to corresponding elements of each tuple in the list, and returns
    tuple of reduction function outputs with length matching the length of tuple
    val_list[0]. It is assumed that all tuples in the list have the same length
    and red_func can be applied to all elements in each corresponding position.
    """
    if isinstance(val_list[0], torch.Tensor):
        return red_func(val_list)
    elif isinstance(val_list[0], bool):
        return any(val_list)
    elif isinstance(val_list[0], tuple):
        final_out = []
        for i in range(len(val_list[0])):
            final_out.append(
                _reduce_list([val_elem[i] for val_elem in val_list], red_func)
            )
    else:
        raise AssertionError(
            "Elements to be reduced can only be"
            "either Tensors or tuples containing Tensors."
        )
    return tuple(final_out)


def _sort_key_list(
    keys: List[device], device_ids: Union[None, List[int]] = None
) -> List[device]:
    """
    Sorts list of torch devices (keys) by given index list, device_ids. If keys
    contains only one device, then the list is returned unchanged. If keys
    contains a device for which the id is not contained in device_ids, then
    an error is returned. This method is used to identify the order of DataParallel
    batched devices, given the device ID ordering.
    """
    if len(keys) == 1:
        return keys
    id_dict: Dict[int, device] = {}
    assert device_ids is not None, "Device IDs must be provided with multiple devices."
    for key in keys:
        if key.index in id_dict:
            raise AssertionError("Duplicate CUDA Device ID identified in device list.")
        id_dict[key.index] = key

    out_list = [
        id_dict[device_id]
        for device_id in filter(lambda device_id: device_id in id_dict, device_ids)
    ]

    assert len(out_list) == len(keys), "Given Device ID List does not match"
    "devices with computed tensors."

    return out_list


def _batched_generator(
    inputs: TensorOrTupleOfTensorsGeneric,
    additional_forward_args: Any = None,
    target_ind: TargetType = None,
    internal_batch_size: Union[None, int] = None,
) -> Iterator[Tuple[Tuple[Tensor, ...], Any, TargetType]]:
    """
    Returns a generator which returns corresponding chunks of size internal_batch_size
    for both inputs and additional_forward_args. If batch size is None,
    generator only includes original inputs and additional args.
    """
    assert internal_batch_size is None or (
        isinstance(internal_batch_size, int) and internal_batch_size > 0
    ), "Batch size must be greater than 0."
    inputs = _format_input(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    num_examples = inputs[0].shape[0]
    if internal_batch_size is None:
        yield inputs, additional_forward_args, target_ind
    else:
        for current_total in range(0, num_examples, internal_batch_size):
            yield _tuple_splice_range(
                inputs, current_total, current_total + internal_batch_size
            ), _tuple_splice_range(
                additional_forward_args,
                current_total,
                current_total + internal_batch_size,
            ), target_ind[
                current_total : current_total + internal_batch_size
            ] if isinstance(
                target_ind, list
            ) or (
                isinstance(target_ind, torch.Tensor) and target_ind.numel() > 1
            ) else target_ind


def _batched_operator(
    operator: Callable[..., TupleOrTensorOrBoolGeneric],
    inputs: TensorOrTupleOfTensorsGeneric,
    additional_forward_args: Any = None,
    target_ind: TargetType = None,
    internal_batch_size: Union[None, int] = None,
    **kwargs: Any
) -> TupleOrTensorOrBoolGeneric:
    """
    Batches the operation of the given operator, applying the given batch size
    to inputs and additional forward arguments, and returning the concatenation
    of the results of each batch.
    """
    all_outputs = [
        operator(
            inputs=input,
            additional_forward_args=additional,
            target_ind=target,
            **kwargs
        )
        for input, additional, target in _batched_generator(
            inputs, additional_forward_args, target_ind, internal_batch_size
        )
    ]
    return _reduce_list(all_outputs)
