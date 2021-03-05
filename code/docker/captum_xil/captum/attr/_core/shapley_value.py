#!/usr/bin/env python3

import itertools
import warnings
from typing import Any, Callable, Iterable, Sequence, Tuple, Union

import torch
from torch import Tensor

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
    _tensorize_baseline,
)
from .._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric


def _all_perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for perm in itertools.permutations(range(num_features)):
        yield perm


def _perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for _ in range(num_samples):
        yield torch.randperm(num_features).tolist()


class ShapleyValueSampling(PerturbationAttribution):
    """
    A perturbation based approach to compute attribution, based on the concept
    of Shapley Values from cooperative game theory. This method involves taking
    a random permutation of the input features and adding them one-by-one to the
    given baseline. The output difference after adding each feature corresponds
    to its attribution, and these difference are averaged when repeating this
    process n_samples times, each time choosing a new random permutation of
    the input features.

    By default, each scalar value within
    the input tensors are taken as a feature and added independently. Passing
    a feature mask, allows grouping features to be added together. This can
    be used in cases such as images, where an entire segment or region
    can be grouped together, measuring the importance of the segment
    (feature group). Each input scalar in the group will be given the same
    attribution value equal to the change in output as a result of adding back
    the entire feature group.

    More details regarding Shapley Value sampling can be found in these papers:
    https://www.sciencedirect.com/science/article/pii/S0305054808000804
    https://pdfs.semanticscholar.org/7715/bb1070691455d1fcfc6346ff458dbca77b2c.pdf
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it. The forward function can either
                        return a scalar per example, or a single scalar for the
                        full batch. If a single scalar is returned for the batch,
                        `perturbations_per_eval` must be 1, and the returned
                        attributions will have first dimension 1, corresponding to
                        feature importance across all examples in the batch.
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.permutation_generator = _perm_generator

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
    ) -> TensorOrTupleOfTensorsGeneric:
        r""""
        NOTE: The feature_mask argument differs from other perturbation based
        methods, since feature indices can overlap across tensors. See the
        description of the feature_mask argument below for more details.

        Args:

                inputs (tensor or tuple of tensors):  Input for which Shapley value
                            sampling attributions are computed. If forward_func takes
                            a single tensor as input, a single input tensor should
                            be provided.
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
                            which difference is computed (for classification cases,
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
                            features which should be added together. feature_mask
                            should contain the same number of tensors as inputs.
                            Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Values across
                            all tensors should be integers in the range 0 to
                            num_features - 1, and indices corresponding to the same
                            feature should have the same value.
                            Note that features are grouped across tensors
                            (unlike feature ablation and occlusion), so
                            if the same index is used in different tensors, those
                            features are still grouped and added simultaneously.
                            If the forward function returns a single scalar per batch,
                            we enforce that the first dimension of each mask must be 1,
                            since attributions are returned batch-wise rather than per
                            example, so the attributions must correspond to the
                            same features (indices) in each input example.
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature
                            Default: None
                n_samples (int, optional):  The number of feature permutations
                            tested.
                            Default: `25` if `n_samples` is not provided.
                perturbations_per_eval (int, optional): Allows multiple ablations
                            to be processed simultaneously in one call to forward_fn.
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
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining ShapleyValueSampling interpreter
            >>> svs = ShapleyValueSampling(net)
            >>> # Computes attribution, taking random orderings
            >>> # of the 16 features and computing the output change when adding
            >>> # each feature. We average over 200 trials (random permutations).
            >>> attr = svs.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we may want to add features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and adding them together.
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
            >>> # With this mask, all inputs with the same value are added
            >>> # together, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = svs.attribute(input, target=1, feature_mask=feature_mask)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        feature_mask = _format_input(feature_mask) if feature_mask is not None else None
        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Ablations per evaluation must be at least 1."

        with torch.no_grad():
            baselines = _tensorize_baseline(inputs, baselines)
            num_examples = inputs[0].shape[0]

            if feature_mask is None:
                feature_mask, total_features = self.construct_feature_mask(inputs)
            else:
                total_features = int(
                    max(torch.max(single_mask).item() for single_mask in feature_mask)
                    + 1
                )
            initial_eval = _run_forward(
                self.forward_func, baselines, target, additional_forward_args
            )
            agg_output_mode = _find_output_mode_and_verify(
                initial_eval, num_examples, perturbations_per_eval, feature_mask
            )

            # Initialize attribution totals and counts
            total_attrib = [
                torch.zeros_like(input[0:1] if agg_output_mode else input)
                for input in inputs
            ]

            iter_count = 0
            # Iterate for number of samples, generate a permutation of the features
            # and evalute the incremental increase for each feature.
            for feature_permutation in self.permutation_generator(
                total_features, n_samples
            ):
                iter_count += 1
                prev_results = initial_eval
                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_masks,
                ) in self._perturbation_generator(
                    inputs,
                    additional_forward_args,
                    target,
                    baselines,
                    feature_mask,
                    feature_permutation,
                    perturbations_per_eval,
                ):
                    if sum(torch.sum(mask).item() for mask in current_masks) == 0:
                        warnings.warn(
                            "Feature mask is missing some integers between 0 and "
                            "num_features, for optimal performance, make sure each"
                            " consecutive integer corresponds to a feature."
                        )
                    # modified_eval dimensions: 1D tensor with length
                    # equal to #num_examples * #features in batch
                    modified_eval = _run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )
                    if agg_output_mode:
                        eval_diff = modified_eval - prev_results
                        prev_results = modified_eval
                    else:
                        all_eval = torch.cat((prev_results, modified_eval), dim=0)
                        eval_diff = all_eval[num_examples:] - all_eval[:-num_examples]
                        prev_results = all_eval[-num_examples:]
                    for j in range(len(total_attrib)):
                        current_eval_diff = eval_diff
                        if not agg_output_mode:
                            # current_eval_diff dimensions:
                            # (#features in batch, #num_examples, 1,.. 1)
                            # (contains 1 more dimension than inputs). This adds extra
                            # dimensions of 1 to make the tensor broadcastable with the
                            # inputs tensor.
                            current_eval_diff = current_eval_diff.reshape(
                                (-1, num_examples) + (len(inputs[j].shape) - 1) * (1,)
                            )
                        total_attrib[j] += (
                            current_eval_diff * current_masks[j].float()
                        ).sum(dim=0)

            # Divide total attributions by number of random permutations and return
            # formatted attributions.
            attrib = tuple(
                tensor_attrib_total / iter_count for tensor_attrib_total in total_attrib
            )
            formatted_attr = _format_attributions(is_inputs_tuple, attrib)
        return formatted_attr

    def _perturbation_generator(
        self,
        inputs: Tuple[Tensor, ...],
        additional_args: Any,
        target: TargetType,
        baselines: Tuple[Tensor, ...],
        input_masks: TensorOrTupleOfTensorsGeneric,
        feature_permutation: Sequence[int],
        perturbations_per_eval: int,
    ) -> Iterable[Tuple[Tuple[Tensor, ...], Any, TargetType, Tuple[Tensor, ...]]]:
        """
        This method is a generator which yields each perturbation to be evaluated
        including inputs, additional_forward_args, targets, and mask.
        """
        # current_tensors starts at baselines and includes each additional feature as
        # added based on the permutation order.
        current_tensors = baselines
        current_tensors_list = []
        current_mask_list = []

        # Compute repeated additional args and targets
        additional_args_repeated = (
            _expand_additional_forward_args(additional_args, perturbations_per_eval)
            if additional_args is not None
            else None
        )
        target_repeated = _expand_target(target, perturbations_per_eval)
        for i in range(len(feature_permutation)):
            current_tensors = tuple(
                current
                * (torch.tensor(1) - (mask == feature_permutation[i]).to(current.dtype))
                + input * (mask == feature_permutation[i]).to(input.dtype)
                for input, current, mask in zip(inputs, current_tensors, input_masks)
            )
            current_tensors_list.append(current_tensors)
            current_mask_list.append(
                tuple(mask == feature_permutation[i] for mask in input_masks)
            )
            if len(current_tensors_list) == perturbations_per_eval:
                combined_inputs = tuple(
                    torch.cat(aligned_tensors, dim=0)
                    for aligned_tensors in zip(*current_tensors_list)
                )
                combined_masks = tuple(
                    torch.stack(aligned_masks, dim=0)
                    for aligned_masks in zip(*current_mask_list)
                )
                yield (
                    combined_inputs,
                    additional_args_repeated,
                    target_repeated,
                    combined_masks,
                )
                current_tensors_list = []
                current_mask_list = []

        # Create batch with remaining evaluations, may not be a complete batch
        # (= perturbations_per_eval)
        if len(current_tensors_list) != 0:
            additional_args_repeated = (
                _expand_additional_forward_args(
                    additional_args, len(current_tensors_list)
                )
                if additional_args is not None
                else None
            )
            target_repeated = _expand_target(target, len(current_tensors_list))
            combined_inputs = tuple(
                torch.cat(aligned_tensors, dim=0)
                for aligned_tensors in zip(*current_tensors_list)
            )
            combined_masks = tuple(
                torch.stack(aligned_masks, dim=0)
                for aligned_masks in zip(*current_mask_list)
            )
            yield (
                combined_inputs,
                additional_args_repeated,
                target_repeated,
                combined_masks,
            )

    def construct_feature_mask(
        self, inputs: Tuple[Tensor, ...]
    ) -> Tuple[Tuple[Tensor, ...], int]:
        feature_mask = []
        current_num_features = 0
        for i in range(len(inputs)):
            num_features = torch.numel(inputs[i][0])
            feature_mask.append(
                current_num_features
                + torch.reshape(
                    torch.arange(num_features, device=inputs[i].device),
                    inputs[i][0:1].shape,
                )
            )
            current_num_features += num_features
        total_features = current_num_features
        feature_mask = tuple(feature_mask)
        return feature_mask, total_features


class ShapleyValues(PerturbationAttribution):
    """
    A perturbation based approach to compute attribution, based on the concept
    of Shapley Values from cooperative game theory. This method involves taking
    each permutation of the input features and adding them one-by-one to the
    given baseline. The output difference after adding each feature corresponds
    to its attribution, and these difference are averaged over all possible
    random permutations of the input features.

    By default, each scalar value within
    the input tensors are taken as a feature and added independently. Passing
    a feature mask, allows grouping features to be added together. This can
    be used in cases such as images, where an entire segment or region
    can be grouped together, measuring the importance of the segment
    (feature group). Each input scalar in the group will be given the same
    attribution value equal to the change in output as a result of adding back
    the entire feature group.

    More details regarding Shapley Values can be found in these papers:
    https://apps.dtic.mil/dtic/tr/fulltext/u2/604084.pdf
    https://www.sciencedirect.com/science/article/pii/S0305054808000804
    https://pdfs.semanticscholar.org/7715/bb1070691455d1fcfc6346ff458dbca77b2c.pdf

    NOTE: The method implemented here is very computationally intensive, and
    should only be used with a very small number of features (e.g. < 7).
    This implementation simply extends ShapleyValueSampling and
    evaluates all permutations, leading to a total of n * n! evaluations for n
    features. Shapley values can alternatively be computed with only 2^n
    evaluations, and we plan to add this approach in the future.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it. The forward function can either
                        return a scalar per example, or a single scalar for the
                        full batch. If a single scalar is returned for the batch,
                        `perturbations_per_eval` must be 1, and the returned
                        attributions will have first dimension 1, corresponding to
                        feature importance across all examples in the batch.
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.shapley_sampling = ShapleyValueSampling(forward_func)
        self.shapley_sampling.permutation_generator = _all_perm_generator

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        perturbations_per_eval: int = 1,
    ) -> TensorOrTupleOfTensorsGeneric:
        r""""
        NOTE: The feature_mask argument differs from other perturbation based
        methods, since feature indices can overlap across tensors. See the
        description of the feature_mask argument below for more details.

        Args:

                inputs (tensor or tuple of tensors):  Input for which Shapley value
                            sampling attributions are computed. If forward_func takes
                            a single tensor as input, a single input tensor should
                            be provided.
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
                            which difference is computed (for classification cases,
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
                            features which should be added together. feature_mask
                            should contain the same number of tensors as inputs.
                            Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Values across
                            all tensors should be integers in the range 0 to
                            num_features - 1, and indices corresponding to the same
                            feature should have the same value.
                            Note that features are grouped across tensors
                            (unlike feature ablation and occlusion), so
                            if the same index is used in different tensors, those
                            features are still grouped and added simultaneously.
                            If the forward function returns a single scalar per batch,
                            we enforce that the first dimension of each mask must be 1,
                            since attributions are returned batch-wise rather than per
                            example, so the attributions must correspond to the
                            same features (indices) in each input example.
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature
                            Default: None
                perturbations_per_eval (int, optional): Allows multiple ablations
                            to be processed simultaneously in one call to forward_fn.
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
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)

            >>> # We may want to add features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and adding them together.
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
            >>> # With this mask, all inputs with the same value are added
            >>> # together, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # With only 4 features, it is feasible to compute exact
            >>> # Shapley Values. These can be computed as follows:
            >>> sv = ShapleyValues(net)
            >>> attr = sv.attribute(input, target=1, feature_mask=feature_mask)
        """
        if feature_mask is None:
            total_features = sum(torch.numel(inp[0]) for inp in _format_input(inputs))
        else:
            total_features = (
                int(max(torch.max(single_mask).item() for single_mask in feature_mask))
                + 1
            )

        if total_features >= 10:
            warnings.warn(
                "You are attempting to compute Shapley Values with at least 10 "
                "features, which will likely be very computationally expensive."
                "Consider using Shapley Value Sampling instead."
            )

        return self.shapley_sampling.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
        )
