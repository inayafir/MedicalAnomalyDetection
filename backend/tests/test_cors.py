class TestCORS:
    def test_cors_headers_on_health(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:7860",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_cors_get_request(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:7860"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") is not None
