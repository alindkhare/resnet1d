import asyncio
import json

import uvicorn

import ray
from ray.experimental.async_api import _async_init
from ray.experimental.serve.constants import (HTTP_ROUTER_CHECKER_INTERVAL_S,
                                              SERVE_PROFILE_PATH)
from ray.experimental.serve.context import TaskContext
from ray.experimental.serve.utils import BytesEncoder
from urllib.parse import parse_qs
import os
from ray import cloudpickle as pickle
import time
import torch


class JSONResponse:
    """ASGI compliant response class.
    It is expected to be called in async context and pass along
    `scope, receive, send` as in ASGI spec.
    >>> await JSONResponse({"k": "v"})(scope, receive, send)
    """

    def __init__(self, content=None, status_code=200):
        """Construct a JSON HTTP Response.
        Args:
            content (optional): Any JSON serializable object.
            status_code (int, optional): Default status code is 200.
        """
        self.body = self.render(content)
        self.status_code = status_code
        self.raw_headers = [[b"content-type", b"application/json"]]

    def render(self, content):
        if content is None:
            return b""
        if isinstance(content, bytes):
            return content
        return json.dumps(content, cls=BytesEncoder, indent=2).encode()

    async def __call__(self, scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        })
        await send({"type": "http.response.body", "body": self.body})


class HTTPProxy:
    """
    This class should be instantiated and ran by ASGI server.
    >>> import uvicorn
    >>> uvicorn.run(HTTPProxy(kv_store_actor_handle, router_handle))
    # blocks forever
    """

    def __init__(self, handle, num_queries):
        assert ray.is_initialized()
        self.profile_file = open(
            os.environ.get("HEALTH_PROFILE_PATH", "/tmp/serve_profile.jsonl"),
            "w")
        self.handle = handle
        self.num_queries = num_queries
        self.required_data = []
        self.lock = asyncio.Lock()

    async def handle_lifespan_message(self, scope, receive, send):
        assert scope["type"] == "lifespan"

        message = await receive()
        if message["type"] == "lifespan.startup":
            await _async_init()
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})

    async def receive_http_body(self, scope, receive, send):
        body_buffer = []
        more_body = True
        while more_body:
            message = await receive()
            assert message["type"] == "http.request"

            more_body = message["more_body"]
            body_buffer.append(message["body"])

        return b"".join(body_buffer)

    async def append(self, data):
        async with self.lock:
            self.required_data.append(torch.tensor([[data]]))
            length = len(self.required_data)
            data = self.required_data
        return length, data

    def get_tensor(self):

        input_tensor = torch.cat(self.required_data, dim=1)
        input_tensor = torch.stack([input_tensor])
        self.required_data = []
        return input_tensor

    async def __call__(self, scope, receive, send):
        # NOTE: This implements ASGI protocol specified in
        #       https://asgi.readthedocs.io/en/latest/specs/index.html

        if scope["type"] == "lifespan":
            await self.handle_lifespan_message(scope, receive, send)
            return

        assert scope["type"] == "http"
        current_path = scope["path"]
        if current_path == "/":
            request_sent_time = time.time()
            query_string = scope["query_string"].decode("ascii")
            query_kwargs = parse_qs(query_string)
            data = float(query_kwargs.pop("data", [0.])[0])
            length, curr_list = await self.append(data)
            if length == self.num_queries:
                input_tensor = torch.cat(curr_list, dim=1)
                input_tensor = torch.stack([input_tensor])
                async with lock:
                    if len(self.required_data) > length:
                        self.required_data = self.required_data[length:]
                    else:
                        self.required_data = []
                result = await self.handle.remote(data=input_tensor)
            else:
                result = "Value stored!"
            result_received_time = time.time()
            self.profile_file.write(
                json.dumps({
                    "start": request_sent_time,
                    "end": result_received_time
                }))
            self.profile_file.write("\n")
            self.profile_file.flush()
            if isinstance(result, ray.exceptions.RayTaskError):
                await JSONResponse({
                    "error": "internal error, please use python API to debug"
                })(scope, receive, send)
            else:
                await JSONResponse({"result": result})(scope, receive, send)


@ray.remote
class HTTPActor:
    def __init__(self, handle, num_queries):
        self.app = HTTPProxy(handle, num_queries)

    def run(self, host="0.0.0.0", port=5000):
        uvicorn.run(
            self.app, host=host, port=port, lifespan="on", access_log=False)
