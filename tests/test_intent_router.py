import unittest

from app.intent_router import IntentType, classify_intent


class _DummySession:
    def __init__(self, response: str | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc

    def classify_intent_with_llm(self, user_input: str) -> str:
        if self.exc is not None:
            raise self.exc
        return self.response or '{"intent":"CHAT","confidence":0.5,"reason":"ok"}'


class IntentRouterTests(unittest.TestCase):
    def test_classify_intent_falls_back_to_chat_when_llm_unavailable(self):
        session = _DummySession(exc=RuntimeError("connection error"))

        result = classify_intent("查詢 2026年1月 平均信用分", session)

        self.assertEqual(result.intent, IntentType.CHAT)
        self.assertAlmostEqual(result.confidence, 0.2)
        self.assertIn("unavailable", result.reason)

    def test_classify_intent_still_honors_exit_keyword_without_llm(self):
        session = _DummySession(exc=RuntimeError("should not be called"))

        result = classify_intent("exit", session)

        self.assertEqual(result.intent, IntentType.EXIT)
        self.assertEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
