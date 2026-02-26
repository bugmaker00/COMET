# -*- coding: utf-8 -*-
import json
import os
import shutil
import unittest

import numpy as np
import torch

from comet import download_model, load_from_checkpoint
from comet.models import XCOMETMetric
from tests.data import DATA_PATH

with open(f"{DATA_PATH}/expected_outputs.json") as fr:
    TEST_SAMPLES = json.load(fr)

class BaseOutputConsistencyUnifiedMetric(unittest.TestCase):
    """ Detect UnifiedMetric output changes caused by COMET updates. """
    model_name = None
    referenceless = False

    @classmethod
    def setUpClass(cls):
        if cls is BaseOutputConsistencyUnifiedMetric:
            raise unittest.SkipTest("Base class must not be executed.")

        cls.model = load_from_checkpoint(
            download_model(cls.model_name, saving_directory=DATA_PATH)
        )
        cls.gpus = 1 if torch.cuda.device_count() > 0 else 0

    def test_predict(self):
        if self.referenceless:
            self.model_name = f"{self.model_name}_referenceless"

        test_samples = TEST_SAMPLES[self.model_name]

        if self.referenceless:
            test_samples = [{k: v for k, v in sample.items() if k != "ref"} for sample in test_samples]

        model_output = self.model.predict(
            test_samples,
            batch_size=12,
            gpus=self.gpus
        )

        assert "error_spans" in model_output.metadata
        assert "src_scores" in model_output.metadata
        
        if not self.referenceless:
            assert "ref_scores" in model_output.metadata
            assert "unified_scores" in model_output.metadata

        # Check every expected score
        score_types = ["score", "src_score", "mqm_score"]

        if not self.referenceless:
            score_types += ["ref_score", "unified_score"]

        for score_type in score_types:
            expected_scores = np.array(
                [sample[score_type] for sample in test_samples]
            )

            if score_type == "score":
                actual_scores = np.array(model_output.scores)
            else:
                actual_scores = np.array(
                    model_output.metadata[f"{score_type}s"]
                )

            np.testing.assert_almost_equal(
                expected_scores,
                actual_scores,
                decimal=5
            )

            if score_type == "score":
                np.testing.assert_almost_equal(
                    expected_scores.mean(),
                    model_output.system_score,
                    decimal=5
                )

        # Check error spans
        expected_error_spans = [sample["error_spans"] for sample in test_samples]
        self.assertEqual(expected_error_spans, model_output.metadata["error_spans"])


class TestXCOMETXLQE(BaseOutputConsistencyUnifiedMetric):
    model_name = "Unbabel/XCOMET-XL"
    referenceless = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--XCOMET-XL"))


class TestXCOMETXL(BaseOutputConsistencyUnifiedMetric):
    model_name = "Unbabel/XCOMET-XL"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--XCOMET-XL"))

class TestXCOMETXXL(BaseOutputConsistencyUnifiedMetric):
    model_name = "Unbabel/XCOMET-XXL"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--XCOMET-XXL"))


class TestXCOMETXXLQE(BaseOutputConsistencyUnifiedMetric):
    model_name = "Unbabel/XCOMET-XXL"
    referenceless = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--XCOMET-XXL"))


class BaseOutputConsistencyRegressionMetric(unittest.TestCase):
    """ Detect RegressionMetric output changes caused by COMET updates. """

    model_name = None
    referenceless = False

    @classmethod
    def setUpClass(cls):
        if cls is BaseOutputConsistencyRegressionMetric:
            raise unittest.SkipTest("Base class must not be executed.")

        cls.model = load_from_checkpoint(
            download_model(cls.model_name, saving_directory=DATA_PATH)
        )
        cls.gpus = 1 if torch.cuda.device_count() > 0 else 0

    def test_predict(self):
        if self.referenceless:
            self.model_name = f"{self.model_name}_referenceless"

        test_samples = TEST_SAMPLES[self.model_name]

        if self.referenceless:
            test_samples = [{k: v for k, v in sample.items() if k != "ref"} for sample in test_samples]

        model_output = self.model.predict(
            test_samples,
            batch_size=12,
            gpus=self.gpus
        )

        assert "scores" in model_output
        assert "system_score" in model_output

        # Check scores
        expected_scores = np.array([sample["score"] for sample in test_samples])
        np.testing.assert_almost_equal(
            expected_scores,
            model_output.scores,
            decimal=5
        )
        np.testing.assert_almost_equal(
            expected_scores.mean(),
            model_output.system_score,
            decimal=5
        )


class TestWMT22CometDA(BaseOutputConsistencyRegressionMetric):
    model_name = "Unbabel/wmt22-comet-da"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--wmt22-comet-da"))


class TestWMT22CometKiwiDA(BaseOutputConsistencyRegressionMetric):
    model_name = "Unbabel/wmt22-cometkiwi-da"
    referenceless = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--wmt22-cometkiwi-da"))


class TestWMT23CometKiwiDA(BaseOutputConsistencyRegressionMetric):
    model_name = "Unbabel/wmt23-cometkiwi-da-xl"
    referenceless = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(DATA_PATH, "models--Unbabel--wmt23-cometkiwi-da-xl"))
