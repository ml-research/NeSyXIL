#!/usr/bin/env python3

from typing import Any, Callable, Tuple, Union, cast

import torch
from torch import Tensor, dtype

from .._utils.attribution import PerturbationAttribution
from .._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _find_output_mode_and_verify,
    _format_additional_forward_args,
    _format_attributions,
    _format_input,
    _format_input_baseline,
    _is_tuple,
    _run_forward,
)
from .._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric


class FeatureAblation(PerturbationAttribution):
    r"""
    A perturbation based approach to computing attribution, involving
    replacing each input feature with a given baseline / reference, and
    computing the difference in output. By default, each scalar value within
    each input tensor is taken as a feature and replaced independently. Passing
    a feature mask, allows grouping features to be ablated together. This can
    be used in cases such as images, where an entire segment or region
    can be ablated, measuring the importance of the segment (feature group).
    Each input scalar in the group will be given the same attribution value
    equal to the change in target as a result of ablating the entire feature
    group.

    The forward function can either return a scalar per example, or a single
    scalar for the full batch. If a single scalar is returned for the batch,
    `perturbations_per_eval` must be 1, and the returned attributions will have
    first dimension 1, corresponding to feature importance across all
    examples in the batch.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.use_weights = False

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        perturbations_per_eval: int = 1,
        **kwargs: Any
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which ablation
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define reference value which replaces each
                        feature when ablated.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or
                          broadcastable to match the dimensions of inputs

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
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            feature_mask (tensor or tuple of tensors, optional):
                        feature_mask defines a mask for the input, grouping
                        features which should be ablated together. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Each tensor
                        should contain integers in the range 0 to num_features
                        - 1, and indices corresponding to the same feature should
                        have the same value.
                        Note that features within each input tensor are ablated
                        independently (not across tensors).
                        If the forward function returns a single scalar per batch,
                        we enforce that the first dimension of each mask must be 1,
                        since attributions are returned batch-wise rather than per
                        example, so the attributions must correspond to the
                        same features (indices) in each input example.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature, which
                        is ablated independently.
                        Default: None
            perturbations_per_eval (int, optional): Allows ablation of multiple
                        features to be processed simultaneously in one call to
                        forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            **kwargs (Any, optional): Any additional arguments used by child
                        classes of FeatureAblation (such as Occlusion) to construct
                        ablations. These arguments are ignored when using
                        FeatureAblation directly.
                        Default: None

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        The attributions with respect to each input feature.
                        If the forward function returns
                        a scalar value per example, attributions will be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If the forward function returns a scalar per batch, then
                        attribution tensor(s) will have first dimension 1 and
                        the remaining dimensions will match the input.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple of tensors is provided for inputs, a
                        tuple of corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining FeatureAblation interpreter
            >>> ablator = FeatureAblation(net)
            >>> # Computes ablation attribution, ablating each of the 16
            >>> # scalar input independently.
            >>> attr = ablator.attribute(input, target=1)

            >>> # Alternatively, we may want to ablate features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and ablating them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are ablated
            >>> # simultaneously, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = ablator.attribute(input, target=1, feature_mask=feature_mask)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        num_examples = inputs[0].shape[0]
        feature_mask = _format_input(feature_mask) if feature_mask is not None else None
        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Ablations per evaluation must be at least 1."
        with torch.no_grad():
            # Computes initial evaluation with all features, which is compared
            # to each ablated result.
            initial_eval = _run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )
            agg_output_mode = _find_output_mode_and_verify(
                initial_eval, num_examples, perturbations_per_eval, feature_mask
            )
            if not agg_output_mode:
                initial_eval = initial_eval.reshape(1, num_examples)

            # Initialize attribution totals and counts
            attrib_type = cast(
                dtype,
                initial_eval.dtype
                if isinstance(initial_eval, Tensor)
                else type(initial_eval),
            )
            total_attrib = [
                torch.zeros_like(
                    input[0:1] if agg_output_mode else input, dtype=attrib_type
                )
                for input in inputs
            ]

            # Weights are used in cases where ablations may be overlapping.
            if self.use_weights:
                weights = [
                    torch.zeros_like(input[0:1] if agg_output_mode else input).float()
                    for input in inputs
                ]

            # Iterate through each feature tensor for ablation
            for i in range(len(inputs)):
                # Skip any empty input tensors
                if torch.numel(inputs[i]) == 0:
                    continue
                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_mask,
                ) in self._ablation_generator(
                    i,
                    inputs,
                    additional_forward_args,
                    target,
                    baselines,
                    feature_mask,
                    perturbations_per_eval,
                    **kwargs
                ):
                    # modified_eval dimensions: 1D tensor with length
                    # equal to #num_examples * #features in batch
                    modified_eval = _run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )
                    # eval_diff dimensions: (#features in batch, #num_examples, 1,.. 1)
                    # (contains 1 more dimension than inputs). This adds extra
                    # dimensions of 1 to make the tensor broadcastable with the inputs
                    # tensor.
                    if agg_output_mode:
                        eval_diff = initial_eval - modified_eval
                    else:
                        eval_diff = (
                            initial_eval - modified_eval.reshape(-1, num_examples)
                        ).reshape(
                            (-1, num_examples) + (len(inputs[i].shape) - 1) * (1,)
                        )
                    if self.use_weights:
                        weights[i] += current_mask.float().sum(dim=0)
                    total_attrib[i] += (eval_diff * current_mask.to(attrib_type)).sum(
                        dim=0
                    )

            # Divide total attributions by counts and return formatted attributions
            if self.use_weights:
                attrib = tuple(
                    single_attrib.float() / weight
                    for single_attrib, weight in zip(total_attrib, weights)
                )
            else:
                attrib = tuple(total_attrib)
            _result = _format_attributions(is_inputs_tuple, attrib)
        return _result

    def _ablation_generator(
        self,
        i,
        inputs,
        additional_args,
        target,
        baselines,
        input_mask,
        perturbations_per_eval,
        **kwargs
    ):
        """
        This method is a generator which yields each perturbation to be evaluated
        including inputs, additional_forward_args, targets, and mask.
        """
        extra_args = {}
        for key, value in kwargs.items():
            # For any tuple argument in kwargs, we choose index i of the tuple.
            if isinstance(value, tuple):
                extra_args[key] = value[i]
            else:
                extra_args[key] = value

        input_mask = input_mask[i] if input_mask is not None else None
        min_feature, num_features, input_mask = self._get_feature_range_and_mask(
            inputs[i], input_mask, **extra_args
        )
        num_examples = inputs[0].shape[0]
        perturbations_per_eval = min(perturbations_per_eval, num_features)
        baseline = baselines[i] if isinstance(baselines, tuple) else baselines
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.reshape((1,) + baseline.shape)

        # Repeat features and additional args for batch size.
        all_features_repeated = [
            torch.cat([inputs[j]] * perturbations_per_eval, dim=0)
            for j in range(len(inputs))
        ]
        additional_args_repeated = (
            _expand_additional_forward_args(additional_args, perturbations_per_eval)
            if additional_args is not None
            else None
        )
        target_repeated = _expand_target(target, perturbations_per_eval)

        num_features_processed = min_feature
        while num_features_processed < num_features:
            current_num_ablated_features = min(
                perturbations_per_eval, num_features - num_features_processed
            )

            # Store appropriate inputs and additional args based on batch size.
            if current_num_ablated_features != perturbations_per_eval:
                current_features = [
                    feature_repeated[0 : current_num_ablated_features * num_examples]
                    for feature_repeated in all_features_repeated
                ]
                current_additional_args = (
                    _expand_additional_forward_args(
                        additional_args, current_num_ablated_features
                    )
                    if additional_args is not None
                    else None
                )
                current_target = _expand_target(target, current_num_ablated_features)
            else:
                current_features = all_features_repeated
                current_additional_args = additional_args_repeated
                current_target = target_repeated

            # Store existing tensor before modifying
            original_tensor = current_features[i]
            # Construct ablated batch for features in range num_features_processed
            # to num_features_processed + current_num_ablated_features and return
            # mask with same size as ablated batch. ablated_features has dimension
            # (current_num_ablated_features, num_examples, inputs[i].shape[1:])
            # Note that in the case of sparse tensors, the second dimension
            # may not necessarilly be num_examples and will match the first
            # dimension of this tensor.
            current_reshaped = current_features[i].reshape(
                (current_num_ablated_features, -1) + current_features[i].shape[1:]
            )

            ablated_features, current_mask = self._construct_ablated_input(
                current_reshaped,
                input_mask,
                baseline,
                num_features_processed,
                num_features_processed + current_num_ablated_features,
                **extra_args
            )

            # current_features[i] has dimension
            # (current_num_ablated_features * num_examples, inputs[i].shape[1:]),
            # which can be provided to the model as input.
            current_features[i] = ablated_features.reshape(
                (-1,) + ablated_features.shape[2:]
            )
            yield tuple(
                current_features
            ), current_additional_args, current_target, current_mask
            # Replace existing tensor at index i.
            current_features[i] = original_tensor
            num_features_processed += current_num_ablated_features

    def _construct_ablated_input(
        self, expanded_input, input_mask, baseline, start_feature, end_feature, **kwargs
    ):
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines. expanded_input shape is (`num_features`, `num_examples`, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and `num_features` = `end_feature` - `start_feature`.
        input_mask has same number of dimensions as original input tensor (one less
        than `expanded_input`), and can have first dimension either 1, applying same
        feature mask to all examples, or `num_examples`. baseline is expected to
        be broadcastable to match `expanded_input`.

        This method returns the ablated input tensor, which has the same
        dimensionality as `expanded_input` as well as the corresponding mask with
        either the same dimensionality as `expanded_input` or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        current_mask = torch.stack(
            [input_mask == j for j in range(start_feature, end_feature)], dim=0
        ).long()
        ablated_tensor = (
            expanded_input * (1 - current_mask).to(expanded_input.dtype)
        ) + (baseline * current_mask.to(expanded_input.dtype))
        return ablated_tensor, current_mask

    def _get_feature_range_and_mask(self, input, input_mask, **kwargs):
        if input_mask is None:
            # Obtain feature mask for selected input tensor, matches size of
            # 1 input example, (1 x inputs[i].shape[1:])
            input_mask = torch.reshape(
                torch.arange(torch.numel(input[0]), device=input.device),
                input[0:1].shape,
            ).long()
        return (
            torch.min(input_mask).item(),
            torch.max(input_mask).item() + 1,
            input_mask,
        )
