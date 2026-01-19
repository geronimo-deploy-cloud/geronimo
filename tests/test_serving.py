"""Tests for geronimo.serving module."""

import pytest

from geronimo.serving import Endpoint


class TestEndpoint:
    """Tests for Endpoint base class."""

    def test_endpoint_subclass(self):
        """Test creating an endpoint subclass."""
        class TestEndpoint(Endpoint):
            route = "/predict"
            method = "POST"

            def preprocess(self, request):
                return request

            def postprocess(self, result):
                return {"prediction": result}

            def handle(self, request):
                preprocessed = self.preprocess(request)
                # Simulate prediction
                result = sum(preprocessed.get("features", [0]))
                return self.postprocess(result)

        endpoint = TestEndpoint()
        assert endpoint.route == "/predict"
        assert endpoint.method == "POST"

    def test_endpoint_handle(self):
        """Test endpoint request handling."""
        class SumEndpoint(Endpoint):
            route = "/sum"
            method = "POST"

            def preprocess(self, request):
                return request["numbers"]

            def postprocess(self, result):
                return {"sum": result}

            def handle(self, request):
                numbers = self.preprocess(request)
                return self.postprocess(sum(numbers))

        endpoint = SumEndpoint()
        result = endpoint.handle({"numbers": [1, 2, 3, 4, 5]})
        
        assert result["sum"] == 15

    def test_endpoint_with_initialization(self):
        """Test endpoint with model initialization."""
        class ModelEndpoint(Endpoint):
            route = "/predict"
            method = "POST"

            def __init__(self):
                super().__init__()
                self._model = None
                self._initialized = False

            def preprocess(self, request):
                return request

            def postprocess(self, result):
                return result

            def initialize(self):
                # Simulate model loading
                self._model = lambda x: [v * 2 for v in x]
                self._initialized = True

            def handle(self, request):
                if not self._initialized:
                    self.initialize()
                features = request["features"]
                return {"predictions": self._model(features)}

        endpoint = ModelEndpoint()
        result = endpoint.handle({"features": [1, 2, 3]})
        
        assert result["predictions"] == [2, 4, 6]
        assert endpoint._initialized is True
