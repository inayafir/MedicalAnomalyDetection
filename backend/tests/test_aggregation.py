import json

import pytest

from app.aggregation import build_prediction_record
from app.models import CLASSIFIER_CLASSES, DETECTOR_CLASSES


class TestBuildPredictionRecord:
    def test_valid_input_with_classifier_class(self):
        raw = {
            "class": "Cardiomegaly",
            "confidence": 0.85,
            "bboxes": [
                {"class": "Aortic enlargement", "x1": 10, "y1": 20, "x2": 100, "y2": 200, "confidence": 0.8}
            ],
            "heatmap_path": "heatmaps/2026/08/25/test.png",
        }
        result = build_prediction_record(raw, image_id=1)
        assert result["image_id"] == 1
        assert result["predicted_class"] == "Cardiomegaly"
        assert result["confidence"] == 0.85
        assert result["heatmap_path"] == "heatmaps/2026/08/25/test.png"
        assert isinstance(result["bboxes"], str)

    def test_normal_class(self):
        raw = {
            "class": "Normal",
            "confidence": 0.95,
            "bboxes": [],
            "heatmap_path": None,
        }
        result = build_prediction_record(raw, image_id=2)
        assert result["predicted_class"] == "Normal"

    def test_pulmonary_fibrosis(self):
        raw = {
            "class": "Pulmonary fibrosis",
            "confidence": 0.78,
            "bboxes": [
                {"class": "Pleural thickening", "x1": 50, "y1": 60, "x2": 200, "y2": 300, "confidence": 0.75}
            ],
            "heatmap_path": None,
        }
        result = build_prediction_record(raw, image_id=3)
        assert result["predicted_class"] == "Pulmonary fibrosis"

    def test_bbox_uses_detector_class_not_classifier_class(self):
        """Bbox class must be one of the 14 detector classes."""
        raw = {
            "class": "Normal",
            "confidence": 0.95,
            "bboxes": [
                {"class": "Atelectasis", "x1": 10, "y1": 20, "x2": 100, "y2": 200, "confidence": 0.8},
                {"class": "Nodule/Mass", "x1": 50, "y1": 60, "x2": 150, "y2": 180, "confidence": 0.7},
            ],
            "heatmap_path": None,
        }
        result = build_prediction_record(raw, image_id=4)
        bboxes = json.loads(result["bboxes"])
        assert bboxes[0]["class"] == "Atelectasis"
        assert bboxes[1]["class"] == "Nodule/Mass"

    def test_invalid_classifier_class(self):
        raw = {"class": "InvalidClass", "confidence": 0.5, "bboxes": []}
        with pytest.raises(ValueError, match="Invalid classifier class"):
            build_prediction_record(raw, image_id=1)

    def test_invalid_detector_class_in_bbox(self):
        raw = {
            "class": "Cardiomegaly",
            "confidence": 0.8,
            "bboxes": [{"class": "Normal", "x1": 10, "y1": 20, "x2": 100, "y2": 200, "confidence": 0.8}],
        }
        with pytest.raises(ValueError, match="Invalid detector class"):
            build_prediction_record(raw, image_id=1)

    def test_any_classifier_class_accepted(self):
        """All 15 classifier classes should be accepted as top-level class."""
        for cls in CLASSIFIER_CLASSES:
            raw = {"class": cls, "confidence": 0.5, "bboxes": []}
            result = build_prediction_record(raw, image_id=1)
            assert result["predicted_class"] == cls

    def test_confidence_out_of_range(self):
        raw = {"class": "Cardiomegaly", "confidence": 1.5, "bboxes": []}
        with pytest.raises(ValueError, match="Confidence"):
            build_prediction_record(raw, image_id=1)

    def test_bbox_missing_key(self):
        raw = {
            "class": "Cardiomegaly",
            "confidence": 0.8,
            "bboxes": [{"class": "Aortic enlargement", "x1": 10, "y1": 20, "x2": 100}],
        }
        with pytest.raises(ValueError, match="missing required key"):
            build_prediction_record(raw, image_id=1)

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            build_prediction_record("bad", image_id=1)

    def test_bboxes_not_a_list(self):
        raw = {"class": "Cardiomegaly", "confidence": 0.8, "bboxes": "not a list"}
        with pytest.raises(ValueError, match="bboxes must be a list"):
            build_prediction_record(raw, image_id=1)

    def test_normal_with_nonempty_bboxes_accepted(self):
        """Edge case: classifier says Normal but YOLO found regions — valid disagreement."""
        raw = {
            "class": "Normal",
            "confidence": 0.60,
            "bboxes": [
                {"class": "Atelectasis", "x1": 10, "y1": 20, "x2": 100, "y2": 200, "confidence": 0.75},
                {"class": "Calcification", "x1": 50, "y1": 60, "x2": 150, "y2": 180, "confidence": 0.65},
            ],
            "heatmap_path": None,
        }
        result = build_prediction_record(raw, image_id=1)
        assert result["predicted_class"] == "Normal"
        bboxes = json.loads(result["bboxes"])
        assert len(bboxes) == 2
        assert bboxes[0]["class"] == "Atelectasis"
        assert bboxes[1]["class"] == "Calcification"


class TestClassCounts:
    def test_classifier_classes_count(self):
        assert len(CLASSIFIER_CLASSES) == 15

    def test_detector_classes_count(self):
        assert len(DETECTOR_CLASSES) == 14

    def test_detector_is_subset_of_classifier(self):
        assert set(DETECTOR_CLASSES).issubset(set(CLASSIFIER_CLASSES))

    def test_normal_only_in_classifier(self):
        assert "Normal" in CLASSIFIER_CLASSES
        assert "Normal" not in DETECTOR_CLASSES
