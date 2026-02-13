from app.sql_guard import has_metric_name_column_reference


def test_detects_alias_dot_metric_name_reference():
    sql = "SELECT SUM(db.deposit_end_balance) AS x FROM fact_account_balance_daily db"
    selected_metrics = ["deposit_balance_daily.deposit_end_balance"]
    assert has_metric_name_column_reference(sql, selected_metrics)


def test_allows_real_column_expression_for_metric():
    sql = "SELECT SUM(bal.end_balance) AS deposit_balance_daily_deposit_end_balance FROM fact_account_balance_daily bal"
    selected_metrics = ["deposit_balance_daily.deposit_end_balance"]
    assert not has_metric_name_column_reference(sql, selected_metrics)

