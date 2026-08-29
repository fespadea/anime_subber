import concurrent.futures
import threading
import unittest

from anime_subber_core.gemini import GeminiManager


class _FakeClient:
    def __init__(self, failure=None, response=None):
        self.models = self
        self.failure = failure
        self.response = response or object()

    def generate_content(self, **_kwargs):
        if self.failure:
            raise self.failure
        return self.response


class GeminiManagerTests(unittest.TestCase):
    def test_concurrent_lazy_initialization_creates_one_client(self):
        count = 0
        count_lock = threading.Lock()

        def factory():
            nonlocal count
            with count_lock:
                count += 1
            return _FakeClient()

        manager = GeminiManager(client_factory=factory)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            clients = list(executor.map(lambda _: manager.client, range(40)))
        self.assertEqual(count, 1)
        self.assertTrue(all(client is clients[0] for client in clients))

    def test_closed_client_is_recreated_and_retried_once(self):
        expected = object()
        clients = iter([
            _FakeClient(RuntimeError("Cannot send a request, as the client has been closed.")),
            _FakeClient(response=expected),
        ])
        manager = GeminiManager(client_factory=lambda: next(clients))
        response = manager.generate("contents", "config")
        self.assertIs(response, expected)


if __name__ == "__main__":
    unittest.main()
