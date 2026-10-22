import argparse
import json
import time
import urllib.error
import urllib.request
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    task_id = str(uuid4())
    body = json.dumps(
        {
            "samples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
    ).encode()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            f"{args.base_url}/api/v1/tasks",
            data=body,
            headers={"Content-Type": "application/json", "Idempotency-Key": task_id},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                assert response.status == 202
            break
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise
            time.sleep(int(error.headers["Retry-After"]))
    else:
        raise TimeoutError("API did not accept task")
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"{args.base_url}/api/v1/tasks/{task_id}", timeout=3
            ) as response:
                result = json.load(response)
            assert result["result"]["predictions"][0]["class_name"] == "setosa"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            time.sleep(0.1)
    raise TimeoutError("Worker did not complete task")


if __name__ == "__main__":
    main()
