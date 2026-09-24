from importlib.metadata import version as package_version

from httpx import AsyncClient


async def test_version_matches(client: AsyncClient) -> None:
    response = await client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json() == {"version": package_version("mlops")}
