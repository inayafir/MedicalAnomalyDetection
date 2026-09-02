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
        """Bbox class can be any of the 14 detector classes, not just the 5 classifier classes."""
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
        import json
        bboxes = json.loads(result["bboxes"])
        assert bboxes[0]["class"] == "Atelectasis"
        assert bboxes[1]["class"] == "Nodule/Mass"

    def test_invalid_classifier_class(self):
        raw = {"class": "InvalidClass", "confidence": 0.5, "bboxes": []}
        with pytest.raises(ValueError, match="Invalid classifier class"):
            build_prediction_record(raw, image_id=1)

    def test_detector_class_not_validated_as_classifier(self):
        """A detector class used as top-level class should be rejected."""
        raw = {"class": "Atelectasis", "confidence": 0.5, "bboxes": []}
        with pytest.raises(ValueError, match="Invalid classifier class"):
            build_prediction_record(raw, image_id=1)

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
