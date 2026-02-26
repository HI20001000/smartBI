import unittest

from app.llm_service import LLMChatSession


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeClient:
    def __init__(self, content: str | list[str]):
        self._content = content
        self._idx = 0

    def invoke(self, _prompt):
        if isinstance(self._content, list):
            if self._idx < len(self._content):
                value = self._content[self._idx]
                self._idx += 1
            else:
                value = self._content[-1] if self._content else ""
            return _FakeResp(value)
        return _FakeResp(self._content)


class LLMServiceTests(unittest.TestCase):
    def test_extract_sql_text_unwraps_code_fence(self):
        raw = "```sql\nSELECT * FROM t\n```"
        self.assertEqual(LLMChatSession._extract_sql_text(raw), "SELECT * FROM t")

    def test_generate_sql_with_langchain_uses_model_output(self):
        session = object.__new__(LLMChatSession)
        session.client = _FakeClient("SQL: SELECT 1")
        session.settings = type("S", (), {"llm_model": "test-model"})()

        sql = session.generate_sql_with_langchain(
            user_input="test",
            enhanced_plan={
                "selected_dataset_candidates": ["sales"],
                "selected_metrics": [],
                "selected_dimensions": [],
                "selected_filters": [],
            },
            semantic_layer={"datasets": {"sales": {"from": "fact_sales"}}, "entities": {}},
        )

        self.assertEqual(sql, "SELECT 1")

    def test_generate_sql_with_langchain_can_return_trace(self):
        session = object.__new__(LLMChatSession)
        session.client = _FakeClient(["not sql", "```sql\nSELECT 2\n```"])
        session.settings = type("S", (), {"llm_model": "trace-model"})()

        sql, trace = session.generate_sql_with_langchain(
            user_input="test trace",
            enhanced_plan={
                "selected_dataset_candidates": ["sales"],
                "selected_metrics": ["sales.revenue"],
                "selected_dimensions": ["sales.biz_date"],
                "selected_filters": [],
            },
            semantic_layer={"datasets": {"sales": {"from": "fact_sales"}}, "entities": {}},
            return_trace=True,
        )

        self.assertEqual(sql, "SELECT 2")
        self.assertEqual(trace["generator"], "langchain.ChatOpenAI.invoke")
        self.assertEqual(trace["model"], "trace-model")
        self.assertEqual(trace["selected_dataset"], "sales")
        self.assertEqual(trace["final_sql"], "SELECT 2")
        self.assertEqual(trace["attempt_count"], 2)
        self.assertFalse(trace["attempts"][0]["ok"])
        self.assertEqual(trace["attempts"][0]["error"], "INVALID_SQL_SHAPE")
        self.assertTrue(trace["attempts"][1]["ok"])

    def test_generate_sql_with_langchain_fails_after_retries(self):
        session = object.__new__(LLMChatSession)
        session.client = _FakeClient(["bad", "still bad", "also bad"])
        session.settings = type("S", (), {"llm_model": "trace-model"})()

        with self.assertRaisesRegex(ValueError, "valid single SELECT SQL"):
            session.generate_sql_with_langchain(
                user_input="test trace",
                enhanced_plan={
                    "selected_dataset_candidates": ["sales"],
                    "selected_metrics": [],
                    "selected_dimensions": [],
                    "selected_filters": [],
                },
                semantic_layer={"datasets": {"sales": {"from": "fact_sales"}}, "entities": {}},
            )


if __name__ == "__main__":
    unittest.main()
