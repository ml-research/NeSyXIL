#!/usr/bin/env python3

from __future__ import print_function

import unittest

import torch
from torch.nn import Embedding

from captum.attr._models.base import (
    InterpretableEmbeddingBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

from ..helpers.basic_models import BasicEmbeddingModel, TextModule
from ..helpers.utils import assertArraysAlmostEqual


class Test(unittest.TestCase):
    def test_interpretable_embedding_base(self):
        input1 = torch.tensor([2, 5, 0, 1])
        input2 = torch.tensor([3, 0, 0, 2])
        model = BasicEmbeddingModel()
        output = model(input1, input2)
        interpretable_embedding1 = configure_interpretable_embedding_layer(
            model, "embedding1"
        )
        self.assertEqual(model.embedding1, interpretable_embedding1)
        self._assert_embeddings_equal(
            input1,
            output,
            interpretable_embedding1,
            model.embedding1.embedding_dim,
            model.embedding1.num_embeddings,
        )
        interpretable_embedding2 = configure_interpretable_embedding_layer(
            model, "embedding2.inner_embedding"
        )
        self.assertEqual(model.embedding2.inner_embedding, interpretable_embedding2)
        self._assert_embeddings_equal(
            input2,
            output,
            interpretable_embedding2,
            model.embedding2.inner_embedding.embedding_dim,
            model.embedding2.inner_embedding.num_embeddings,
        )
        # configure another embedding when one is already configured
        with self.assertRaises(AssertionError):
            configure_interpretable_embedding_layer(model, "embedding2.inner_embedding")
        with self.assertRaises(AssertionError):
            configure_interpretable_embedding_layer(model, "embedding1")
        # remove interpretable embedding base
        self.assertTrue(
            model.embedding2.inner_embedding.__class__ is InterpretableEmbeddingBase
        )
        remove_interpretable_embedding_layer(model, interpretable_embedding2)
        self.assertTrue(model.embedding2.inner_embedding.__class__ is Embedding)

        self.assertTrue(model.embedding1.__class__ is InterpretableEmbeddingBase)
        remove_interpretable_embedding_layer(model, interpretable_embedding1)
        self.assertTrue(model.embedding1.__class__ is Embedding)

    def test_custom_module(self):
        input1 = torch.tensor([[3, 2, 0], [1, 2, 4]])
        input2 = torch.tensor([[0, 1, 0], [1, 2, 3]])
        model = BasicEmbeddingModel()
        output = model(input1, input2)
        expected = model.embedding2(input=input2)
        # in this case we make interpretable the custom embedding layer - TextModule
        interpretable_embedding = configure_interpretable_embedding_layer(
            model, "embedding2"
        )
        actual = interpretable_embedding.indices_to_embeddings(input=input2)
        output_interpretable_models = model(input1, actual)
        assertArraysAlmostEqual(output, output_interpretable_models)

        # using assertArraysAlmostEqual instead of assertTensorAlmostEqual because
        # it is important and necessary that each element in comparing tensors
        # match exactly.
        assertArraysAlmostEqual(expected, actual, 0.0)
        self.assertTrue(model.embedding2.__class__ is InterpretableEmbeddingBase)
        remove_interpretable_embedding_layer(model, interpretable_embedding)
        self.assertTrue(model.embedding2.__class__ is TextModule)
        self._assert_embeddings_equal(input2, output, interpretable_embedding)

    def test_nested_multi_embeddings(self):
        input1 = torch.tensor([[3, 2, 0], [1, 2, 4]])
        input2 = torch.tensor([[0, 1, 0], [2, 6, 8]])
        input3 = torch.tensor([[4, 1, 0], [2, 2, 8]])
        model = BasicEmbeddingModel(nested_second_embedding=True)
        output = model(input1, input2, input3)
        expected = model.embedding2(input=input2, another_input=input3)
        # in this case we make interpretable the custom embedding layer - TextModule
        interpretable_embedding2 = configure_interpretable_embedding_layer(
            model, "embedding2"
        )
        actual = interpretable_embedding2.indices_to_embeddings(
            input=input2, another_input=input3
        )
        output_interpretable_models = model(input1, actual)
        assertArraysAlmostEqual(output, output_interpretable_models)

        # using assertArraysAlmostEqual instead of assertTensorAlmostEqual because
        # it is important and necessary that each element in comparing tensors
        # match exactly.
        assertArraysAlmostEqual(expected, actual, 0.0)
        self.assertTrue(model.embedding2.__class__ is InterpretableEmbeddingBase)
        remove_interpretable_embedding_layer(model, interpretable_embedding2)
        self.assertTrue(model.embedding2.__class__ is TextModule)
        self._assert_embeddings_equal(input2, output, interpretable_embedding2)

    def _assert_embeddings_equal(
        self,
        input,
        output,
        interpretable_embedding,
        embedding_dim=None,
        num_embeddings=None,
    ):
        if interpretable_embedding.embedding_dim is not None:
            self.assertEqual(embedding_dim, interpretable_embedding.embedding_dim)
            self.assertEqual(num_embeddings, interpretable_embedding.num_embeddings)

        # dim - [4, 100]
        emb_shape = interpretable_embedding.indices_to_embeddings(input).shape
        self.assertEqual(emb_shape[0], input.shape[0])
        if interpretable_embedding.embedding_dim is not None:
            self.assertEqual(emb_shape[1], interpretable_embedding.embedding_dim)
        self.assertEqual(input.shape[0], output.shape[0])
