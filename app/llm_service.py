import json
import re
from decimal import Decimal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.query_executor import SQLQueryExecutor


class LLMChatSession:
    _DIMENSION_HINT_PATTERNS: tuple[tuple[str, str], ...] = (
        (r"姓名", "姓名"),
        (r"電子(?:信箱|郵件)", "電子信箱"),
        (r"(?:郵件|郵箱|電郵|mail)", "電子信箱"),
        (r"(?:電話|手機|聯絡電話|phone|tel)", "電話"),
        (r"(?:聯絡方式|聯絡資料|聯繫方式)", "聯絡方式"),
        (r"\b(?:email|e-mail)\b", "電子信箱"),
    )
    _CUSTOMER_FILTER_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?:客戶|客户|客戶id|客戶ID|customer|client)\s*[:=]\s*([A-Za-z0-9_-]+)", flags=re.IGNORECASE),
        re.compile(r"(?:客戶|客户|客戶id|客戶ID|customer|client)\s+([A-Za-z0-9_-]+)", flags=re.IGNORECASE),
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        self.history = [
            SystemMessage(content="你是個助理，請用繁體中文回答，回答要清楚、簡潔。")
        ]

    def ask(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        try:
            resp = self.client.invoke(self.history)
            reply = getattr(resp, "content", str(resp)).strip()
        except Exception:
            self.history.pop()
            raise

        self.history.append(AIMessage(content=reply))
        return reply

    def classify_intent_with_llm(self, user_input: str) -> str:
        prompt = [
            SystemMessage(
                content=(
                    "你是意圖分類器。請判斷使用者輸入意圖並輸出 JSON。"
                    "可用 intent 僅有 EXIT、SQL、CHAT。"
                    "輸出格式固定為："
                    '{"intent":"CHAT","confidence":0.0,"reason":"..."}'
                    "不要輸出任何 JSON 以外文字。"
                )
            ),
            HumanMessage(content=user_input),
        ]
        resp = self.client.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()

    def normalize_sql_user_input_with_llm(self, user_input: str) -> dict:
        normalized_input = (user_input or "").strip()
        if not normalized_input:
            return {"normalized_input": "", "changed": False}

        prompt = [
            SystemMessage(
                content=(
                    "你是 SmartBI 查詢語句修正器。"
                    "請將使用者查詢中的明顯錯字、全半形、大小寫、空白與常見中英混用詞彙修正成更標準的查詢文字。"
                    "不要改變原始意圖，不要新增未提及的條件。"
                    "只輸出 JSON，格式固定為："
                    '{"normalized_input":"..."}'
                )
            ),
            HumanMessage(content=f"user_input={normalized_input}"),
        ]

        try:
            resp = self.client.invoke(prompt)
            raw = getattr(resp, "content", str(resp)).strip()
            parsed = json.loads(raw)
            candidate = str(parsed.get("normalized_input", "") or "").strip()
            if candidate:
                normalized_input = candidate
        except Exception:
            pass

        return {
            "normalized_input": normalized_input,
            "changed": normalized_input != (user_input or "").strip(),
        }

    def extract_sql_features_with_llm(self, user_input: str) -> dict:
        prompt = [
            SystemMessage(
                content=(
                    "你是 SmartBI 查詢解析器（SQL/BI Query Feature Extractor）。"
                    "任務：從使用者輸入（中文/英文/混合）提取查詢特徵，並且【只能輸出 JSON】。"
                    "JSON 格式固定為："
                    "{\"tokens\":[],\"metrics\":[],\"dimensions\":[],\"filters\":[],\"time_start\":\"\",\"time_end\":\"\"}"
                    "\n\n"
                    "【輸出規則（嚴格遵守）】\n"
                    "1) 只能輸出以上 6 個欄位，不得新增欄位；不得輸出任何 JSON 以外文字。\n"
                    "2) tokens/metrics/dimensions/filters 必須是【字串陣列】；time_start/time_end 必須是字串。\n"
                    "3) time_start/time_end 格式必須為 yyyy-mm-dd；若無法判定則輸出空字串 \"\"。\n"
                    "4) 不要臆測：使用者沒提到的內容不要填；不確定就留空。\n"
                    "5) 去重：陣列內不得重複字串；保持由重要到次要的順序。\n"
                    "6) 若成功解析為具體日期（time_start/time_end 非空），時間詞不要再放入 tokens 或 filters。\n"
                    "\n"
                    "【欄位語義（SmartBI 導向）】\n"
                    "- metrics：可聚合的指標/度量（例如：銷售額、訂單數、GMV、利潤、DAU、轉化率、平均客單價、同比、環比）。\n"
                    "- dimensions：分組/切片維度（例如：日期、月份、地區、省、市、門店、渠道、品類、商品、用戶類型）。\n"
                    "- filters：限制條件，使用【可讀的條件片段】字串，不要求嚴格語法，但要清楚（例如：\"地區=華東\"、\"渠道 in(線上)\"、\"狀態=已支付\"、\"客單價>200\"）。\n"
                    "- tokens：其他關鍵詞（實體、別名、業務名詞、口語詞、主題詞），以及你無法判定是 metrics/dimensions/filters 的重要詞。\n"
                    "\n"
                    "【時間解析規則（優先級最高，必須執行）】\n"
                    "一、明確日期範圍：\n"
                    "- 例如 \"2024-01-01 到 2024-01-31\" => time_start=\"2024-01-01\", time_end=\"2024-01-31\"。\n"
                    "\n"
                    "二、年份：\n"
                    "- \"2024年\" => time_start=\"2024-01-01\", time_end=\"2024-12-31\"。\n"
                    "\n"
                    "三、月份：\n"
                    "- \"2024年1月\" 或 \"2024-01\" => time_start=\"2024-01-01\"，time_end=該月最後一天（需判斷閏年）。\n"
                    "\n"
                    "四、季度：\n"
                    "- \"2024年Q1\" 或 \"2024Q1\" => time_start=\"2024-01-01\", time_end=\"2024-03-31\"。\n"
                    "- Q2/Q3/Q4 依序為 04-01~06-30、07-01~09-30、10-01~12-31。\n"
                    "\n"
                    "五、相對時間（必須基於系統當前日期與系統時區計算，需輸出具體 yyyy-mm-dd）\n"
                    "- 今天 => time_start=today, time_end=today。\n"
                    "- 昨天 => time_start=today-1day, time_end=today-1day。\n"
                    "- 近7天/最近7天 => time_start=today-6days, time_end=today（包含今天共7天）。\n"
                    "- 近N天/最近N天 => time_start=today-(N-1)days, time_end=today。\n"
                    "- 最近一個月 => time_start=today-1month+1day, time_end=today。\n"
                    "- 本月 => time_start=本月第一天, time_end=today。\n"
                    "- 上月 => time_start=上月第一天, time_end=上月最後一天。\n"
                    "- 今年 => time_start=今年1月1日, time_end=today。\n"
                    "- 去年 => time_start=去年1月1日, time_end=去年12月31日。\n"
                    "若同時給出基準日期（例如：以2024-03-10為基準的近7天），則以該日期為基準計算。\n"
                    "只有在完全無法判斷時間範圍時，time_start/time_end 才允許為空字串 \"\"。\n"
                    "\n"
                    "【分類優先級（避免亂放）】\n"
                    "1) 能明確聚合/指標 => metrics\n"
                    "2) 能明確分組/枚舉 => dimensions\n"
                    "3) 明確條件限制（=、>、<、包含、topN、區間、in、between、是否、狀態）=> filters\n"
                    "4) 其餘重要詞 => tokens\n"
                    "\n"
                    "【常見指標詞映射（看見就優先進 metrics）】\n"
                    "- \"多少\"/\"幾\" + 名詞（訂單/用戶/人數/次數）=> 對應計數型 metrics（例如：\"訂單數\"）\n"
                    "- \"平均\"/\"人均\"/\"每\" => 平均類 metrics（例如：\"平均客單價\"、\"人均消費\"）\n"
                    "- \"增長\"/\"同比\"/\"環比\" => 增長類 metrics（例如：\"同比增長率\"）\n"
                    "\n"
                    "輸出要求：最終只輸出 JSON 物件字串（不要 markdown，不要解釋）。"
                )
            ),
            HumanMessage(content=user_input),
        ]

        try:
            resp = self.client.invoke(prompt)
            raw = getattr(resp, "content", str(resp)).strip()
            parsed = json.loads(raw)
        except Exception:
            parsed = {}

        def _string_list(value: object) -> list[str]:
            if not isinstance(value, list):
                return []
            return [v.strip() for v in value if isinstance(v, str) and v.strip()]

        def _date_or_empty(value: object) -> str:
            if not isinstance(value, str):
                return ""
            value = value.strip()
            if len(value) == 10 and value[4] == "-" and value[7] == "-":
                return value
            return ""

        tokens = _string_list(parsed.get("tokens"))
        metrics = _string_list(parsed.get("metrics"))
        dimensions = _string_list(parsed.get("dimensions"))
        filters = _string_list(parsed.get("filters"))

        normalized_input = (user_input or "").strip()
        if not dimensions:
            for pattern, normalized in self._DIMENSION_HINT_PATTERNS:
                if re.search(pattern, normalized_input, flags=re.IGNORECASE) and normalized not in dimensions:
                    dimensions.append(normalized)

        if not filters:
            for pattern in self._CUSTOMER_FILTER_PATTERNS:
                m = pattern.search(normalized_input)
                if not m:
                    continue
                customer_value = (m.group(1) or "").strip()
                if customer_value:
                    filters.append(f"客戶={customer_value}")
                    break

        return {
            "tokens": tokens,
            "metrics": metrics,
            "dimensions": dimensions,
            "filters": filters,
            "time_start": _date_or_empty(parsed.get("time_start")),
            "time_end": _date_or_empty(parsed.get("time_end")),
            "query_text": user_input.strip(),
        }



    @staticmethod
    def _extract_sql_text(raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        upper = text.upper()
        marker = "SQL:"
        idx = upper.find(marker)
        if idx >= 0:
            maybe = text[idx + len(marker):].strip()
            if maybe:
                text = maybe

        return text

    def generate_sql_with_langchain(
        self,
        user_input: str,
        enhanced_plan: dict,
        semantic_layer: dict,
        *,
        return_trace: bool = False,
    ) -> str | tuple[str, dict]:
        datasets = enhanced_plan.get("selected_dataset_candidates", []) or []
        selected_dataset = str(datasets[0]).strip() if datasets else ""
        if not selected_dataset:
            raise ValueError("No dataset candidates available for SQL generation.")

        datasets_map = (semantic_layer or {}).get("datasets", {}) or {}
        dataset = datasets_map.get(selected_dataset, {}) or {}
        if not dataset:
            raise ValueError(f"Dataset not found in semantic layer: {selected_dataset}")

        entities = (semantic_layer or {}).get("entities", {}) or {}
        entity_tables = {name: str((payload or {}).get("table", "") or "").strip() for name, payload in entities.items()}

        system_prompt = (
            "你是 SmartBI SQL 生成器。請只輸出一條可執行的 MySQL SELECT 查詢。"
            "不得輸出任何解釋、markdown、註解或多語句。"
            "必須遵守語意層定義的 from / joins / metrics / dimensions / filters。"
            "禁止選取語意層未允許欄位。"
        )
        human_prompt = (
            f"user_input={user_input}\n"
            f"selected_dataset={selected_dataset}\n"
            f"plan_json={json.dumps(enhanced_plan, ensure_ascii=False)}\n"
            f"dataset_json={json.dumps(dataset, ensure_ascii=False)}\n"
            f"entity_tables_json={json.dumps(entity_tables, ensure_ascii=False)}\n"
            "請依 plan_json 產生 SQL。若有 selected_metrics 就聚合；selected_dimensions 需出現在 SELECT 並對應 GROUP BY。"
            "selected_filters 需完整轉成 WHERE。between 用 BETWEEN，in 用 IN。"
            "僅回傳 SQL 本文。"
        )

        attempts: list[dict] = []
        sql = ""
        max_attempts = 3
        for idx in range(1, max_attempts + 1):
            retry_hint = ""
            if attempts:
                last = attempts[-1]
                retry_hint = (
                    "\n上一輪生成未通過，請僅回傳單條可執行 SELECT SQL。"
                    f"\n上一輪錯誤：{last.get('error', 'UNKNOWN')}"
                    f"\n上一輪輸出：{last.get('raw_response', '')}"
                )

            prompt = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt + retry_hint),
            ]

            try:
                resp = self.client.invoke(prompt)
                raw = getattr(resp, "content", str(resp)).strip()
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": idx,
                        "raw_response": "",
                        "normalized_sql": "",
                        "ok": False,
                        "error": str(exc) or "LLM_INVOKE_FAILED",
                    }
                )
                continue

            normalized = self._extract_sql_text(raw)
            safe_sql = SQLQueryExecutor._normalize_single_select_sql(normalized)
            if not safe_sql:
                attempts.append(
                    {
                        "attempt": idx,
                        "raw_response": raw,
                        "normalized_sql": normalized,
                        "ok": False,
                        "error": "INVALID_SQL_SHAPE",
                    }
                )
                continue

            sql = safe_sql
            attempts.append(
                {
                    "attempt": idx,
                    "raw_response": raw,
                    "normalized_sql": safe_sql,
                    "ok": True,
                    "error": "",
                }
            )
            break

        if not sql:
            raise ValueError("LLM did not return valid single SELECT SQL.")

        trace = {
            "generator": "langchain.ChatOpenAI.invoke",
            "model": self.settings.llm_model,
            "selected_dataset": selected_dataset,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "final_sql": sql,
        }
        if return_trace:
            return sql, trace
        return sql

    def enhance_semantic_selection_with_llm(
        self,
        user_input: str,
        extracted_features: dict,
        token_hits: dict,
    ) -> dict:
        matches = token_hits.get("matches", []) or []
        blocked = token_hits.get("blocked_matches", []) or []
        candidates = (matches + blocked)[:20]
        if not candidates:
            return {}

        prompt = [
            SystemMessage(
                content=(
                    "你是 SmartBI 語意欄位選擇器。"
                    "請根據使用者問題、Step B 特徵與 Step C 候選，補全應選的 metrics/dimensions/datasets。"
                    "只能從 candidate 中挑選 canonical_name；不得臆造。"
                    "僅輸出 JSON："
                    '{"selected_metrics":[],"selected_dimensions":[],"selected_dataset_candidates":[]}'
                )
            ),
            HumanMessage(
                content=(
                    f"user_input={user_input}\n"
                    f"features_json={json.dumps(extracted_features, ensure_ascii=False)}\n"
                    f"candidates_json={json.dumps(candidates, ensure_ascii=False)}\n"
                    "規則：\n"
                    "1) 指標放 selected_metrics，維度/欄位放 selected_dimensions。\n"
                    "2) 若使用者明確要求多個欄位（例如 姓名與電子郵件），可同時選多個 dimensions。\n"
                    "3) dataset 優先選可支援最多 selected_metrics+selected_dimensions 的資料集。\n"
                    "4) 若無把握，保持陣列為空。\n"
                    "只回傳 JSON。"
                )
            ),
        ]

        try:
            resp = self.client.invoke(prompt)
            raw = getattr(resp, "content", str(resp)).strip()
            parsed = json.loads(raw)
        except Exception:
            return {}

        def _string_list(value: object) -> list[str]:
            if not isinstance(value, list):
                return []
            return [v.strip() for v in value if isinstance(v, str) and v.strip()]

        return {
            "selected_metrics": _string_list(parsed.get("selected_metrics")),
            "selected_dimensions": _string_list(parsed.get("selected_dimensions")),
            "selected_dataset_candidates": _string_list(parsed.get("selected_dataset_candidates")),
        }

    def summarize_query_result_with_llm(self, user_input: str, rows: list[dict], max_rows: int = 20) -> str:
        sample_rows = rows[: max(1, int(max_rows))]

        def _json_fallback(value: object) -> object:
            if isinstance(value, Decimal):
                return float(value)
            return str(value)

        prompt = [
            SystemMessage(
                content=(
                    "你是 SmartBI 報表摘要助手。"
                    "請根據使用者問題與查詢結果，輸出 2~4 句繁體中文摘要。"
                    "要求：聚焦關鍵數據、趨勢與可行觀察，不要杜撰資料。"
                    "若資料筆數很少，請直接點出樣本有限。"
                )
            ),
            HumanMessage(
                content=(
                    f"user_input={user_input}\n"
                    f"rows_json={json.dumps(sample_rows, ensure_ascii=False, default=_json_fallback)}"
                )
            ),
        ]

        try:
            resp = self.client.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception as exc:
            return f"（摘要生成失敗：{exc}）"

    def summarize_failure_with_llm(self, user_input: str, failure_message: str) -> str:
        prompt = [
            SystemMessage(
                content=(
                    "你是 SmartBI 錯誤說明助手。"
                    "請把系統錯誤或校驗失敗訊息整理成 2~4 句繁體中文，"
                    "語氣專業、可執行，包含可能原因與下一步建議，"
                    "不要杜撰未提供的系統狀態。"
                )
            ),
            HumanMessage(content=f"user_input={user_input}\nerror={failure_message}"),
        ]

        try:
            resp = self.client.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            return f"查詢流程發生問題：{failure_message}。請先修正上述錯誤後重試。"
