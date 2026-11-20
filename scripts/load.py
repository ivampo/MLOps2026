import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SAMPLE = {"samples": [{"sepal_length": 5.1, "sepal_width": 3.5,
                       "petal_length": 1.4, "petal_width": 0.2}]}
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate JSONL or send concurrent API requests")
    parser.add_argument("mode", choices=["generate", "api"])
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--path", default="examples/load.jsonl")
    parser.add_argument("--poison", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/tasks")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    if not args.url.startswith(("http://", "https://")):
        parser.error("Only HTTP(S) URLs are supported")
    logging.basicConfig(level=logging.INFO)
    if args.mode == "generate":
        with Path(args.path).open("w", encoding="utf-8") as output:
            for index in range(args.count):
                body = {"invalid": True} if args.poison and index == 0 else SAMPLE
                output.write(json.dumps(body) + "\n")
        logger.info("Created %s tasks in %s", args.count, args.path)
    else:
        def send(index):
            request = Request(args.url, data=json.dumps(SAMPLE).encode(),
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=10) as response:  # noqa: S310 - scheme checked above
                    return response.status
            except HTTPError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            counts = Counter(pool.map(send, range(args.count)))
        logger.info("HTTP status counts: %s", dict(counts))


if __name__ == "__main__":
    main()
