import unittest

from app.llm_service import LLMChatSession


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeClient:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _prompt):
        return _FakeResp(self._content)


class LLMServiceTests(unittest.TestCase):
    def test_extract_sql_text_unwraps_code_fence(self):
        raw = "```sql\nSELECT * FROM t\n```"
        self.assertEqual(LLMChatSession._extract_sql_text(raw), "SELECT * FROM t")

    def test_generate_sql_with_langchain_uses_model_output(self):
        session = object.__new__(LLMChatSession)
        session.client = _FakeClient("SQL: SELECT 1")

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


if __name__ == "__main__":
    unittest.main()
