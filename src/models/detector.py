"""Detector wrapper placeholder."""


class Detector:
    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn"):
        self.model_name = model_name

    def predict(self, image):
        # Return empty predictions placeholder
        return []
