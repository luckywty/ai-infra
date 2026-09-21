from typing import Any, List

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from ray.util import ActorPool


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(self, llm_clients: List[LLMClient]):
        self._llm_client_pool = ActorPool(llm_clients)

    def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        if self._llm_client_pool.has_free():
            self._llm_client_pool.submit(
                lambda client, _request_config: client.llm_request.remote(
                    _request_config
                ),
                request_config,
            )

    def get_next_ready(self, block: bool = False) -> List[Any]:
        """返回已完成请求中已就绪的结果。

        

        参数：
        block：是否阻塞等待，直到有结果就绪。

        返回值：
        一个已就绪结果的列表。

        """
        results = []
        if not block:
            while self._llm_client_pool.has_next():
                results.append(self._llm_client_pool.get_next_unordered())
        else:
            while not self._llm_client_pool.has_next():
                pass
            while self._llm_client_pool.has_next():
                results.append(self._llm_client_pool.get_next_unordered())
        return results
