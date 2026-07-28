"""Tests for GeronimoDeployCloudArtifactBackend."""

import pickle
from unittest.mock import MagicMock, patch, ANY

import pytest
from geronimo.artifacts.gdc_backend import GeronimoDeployCloudArtifactBackend
from geronimo.artifacts import ArtifactStore


class TestGeronimoDeployCloudArtifactBackend:
    """Tests for GDC Backend."""

    @pytest.fixture
    def mock_client(self):
        """Mock Geronimo Cloud Client."""
        client = MagicMock()
        client.api_url = "https://api.test"
        client.headers = {"Authorization": "Bearer test"}
        client.token = "test-token"
        return client

    @pytest.fixture
    def backend(self, mock_client):
        """Backend instance with mock client."""
        return GeronimoDeployCloudArtifactBackend(
            project="test-project",
            version="1.0.0",
            client=mock_client
        )

    @patch("geronimo.gdc.http_utils.httpx.Client")
    def test_save_flow(self, mock_httpx, backend):
        """Test save flow: cloud-save -> upload -> confirm."""
        # Setup mocks
        mock_http = mock_httpx.return_value.__enter__.return_value
        
        # 1. cloud-save response
        mock_http.post.return_value.json.return_value = {
            "upload_url": "https://upload.test/123",
            "id": "artifact-123",
            "s3_uri": "s3://bucket/u/p/v/model.pkl"
        }
        
        # Execute
        uri, size = backend.save(
            "model", 
            {"data": 123}, 
            metadata={"type": "model"}
        )
        
        # Verify
        assert uri == "s3://bucket/u/p/v/model.pkl"
        
        # Check calls
        # 1. cloud-save
        mock_http.post.assert_any_call(
            "/v1/artifacts/cloud-save",
            json={
                "project": "test-project",
                "version": "1.0.0",
                "name": "model",
                "size_bytes": ANY,
                "metadata": {"type": "model"},
            }
        )
        
        # 3. confirm
        mock_http.post.assert_any_call(
            "/v1/artifacts/artifact-123/confirm",
            json={"size_bytes": ANY}
        )

    @patch("geronimo.gdc.http_utils.httpx.Client")
    def test_load_flow(self, mock_httpx, backend):
        """Test load flow: cloud-load -> download -> deserialize."""
        mock_http = mock_httpx.return_value.__enter__.return_value
        
        # cloud-load response
        mock_http.post.return_value.json.return_value = {
            "download_url": "https://download.test/123"
        }
        
        # download response
        mock_http.get.return_value.content = pickle.dumps({"data": 123})
        
        # Execute
        # Case 1: Load by name (uses context)
        data = backend.load("model")
        assert data == {"data": 123}
        
        mock_http.post.assert_called_with(
            "/v1/artifacts/cloud-load",
            json={
                "project": "test-project", 
                "version": "1.0.0", 
                "name": "model"
            }
        )
        
        # Case 2: Load by URI
        backend.load("s3://bucket/u/p/v/other.pkl")
        mock_http.post.assert_called_with(
            "/v1/artifacts/cloud-load",
            json={
                "project": "p", 
                "version": "v", 
                "name": "other"
            }
        )

    @patch("geronimo.gdc.http_utils.httpx.Client")
    def test_list(self, mock_httpx, backend):
        """Test list artifacts."""
        mock_http = mock_httpx.return_value.__enter__.return_value
        mock_http.get.return_value.json.return_value = {
            "artifacts": [{"s3_uri": "s3://bucket/a.pkl"}, {"s3_uri": "s3://bucket/b.pkl"}]
        }
        
        uris = backend.list()
        assert len(uris) == 2
        assert "s3://bucket/a.pkl" in uris
        
        mock_http.get.assert_called_with(
            "/v1/artifacts",
            params={"project": "test-project", "version": "1.0.0"}
        )

    @patch("geronimo.gdc.http_utils.httpx.Client")
    def test_delete(self, mock_httpx, backend):
        """Test delete artifact."""
        mock_http = mock_httpx.return_value.__enter__.return_value
        
        # 1. Search response
        mock_http.get.return_value.json.return_value = {
            "artifacts": [{"id": "art-123"}]
        }
        
        # Execute
        backend.delete("model")
        
        # Verify
        # Search
        mock_http.get.assert_called_with(
            "/v1/artifacts",
            params={"project": "test-project", "version": "1.0.0", "name": "model"}
        )
        # Delete
        mock_http.delete.assert_called_with("/v1/artifacts/art-123")


class TestArtifactStoreIntegration:
    """Test ArtifactStore integration with Cloud Backend."""
    
    def test_gdc_backend_selection(self):
        """Test that setting backend='gdc' uses the GDC backend."""
        with patch("geronimo.artifacts.gdc_backend.GeronimoDeployCloudArtifactBackend") as MockBackend:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.save.return_value = ("s3://uri", 100)
            MockBackend.return_value = mock_instance
            
            # Create store
            store = ArtifactStore(
                project="p", 
                version="v", 
                backend="gdc"
            )
            
            # Verify backend init was called
            MockBackend.assert_called_with(project="p", version="v")
            
            # Verify save delegation
            uri = store.save("model", [1, 2, 3])
            assert uri == "s3://uri"
            mock_instance.save.assert_called_once()
            
            arg_name = mock_instance.save.call_args[0][0]
            assert arg_name == "model"

    def test_namespace_parameter(self):
        """Test namespace parameter is supported."""
        client = MagicMock()
        client.token = "test"
        
        backend = GeronimoDeployCloudArtifactBackend(
            project="p",
            version="v",
            namespace="shared",
            client=client
        )
        
        assert backend.namespace == "shared"

    def test_authentication_check(self):
        """Test that operations fail without authentication."""
        client = MagicMock()
        client.token = None  # No token
        
        backend = GeronimoDeployCloudArtifactBackend(
            project="p",
            version="v",
            client=client
        )
        
        with pytest.raises(RuntimeError, match="Not authenticated"):
            backend.save("model", {}, {})
        
        with pytest.raises(RuntimeError, match="Not authenticated"):
            backend.load("model")
        
        with pytest.raises(RuntimeError, match="Not authenticated"):
            backend.list()
        
        with pytest.raises(RuntimeError, match="Not authenticated"):
            backend.delete("model")
