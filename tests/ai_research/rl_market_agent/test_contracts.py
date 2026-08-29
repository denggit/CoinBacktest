from src.ai_research.rl_market_agent.contracts import PortfolioSelectionKey


def test_champion_priority_is_lexicographic():
    a = PortfolioSelectionKey.from_metrics(max_flat_days=3, max_consecutive_losing_days=8, max_drawdown_pct=20, cagr_pct=30, total_return_pct=100)
    b = PortfolioSelectionKey.from_metrics(max_flat_days=4, max_consecutive_losing_days=2, max_drawdown_pct=5, cagr_pct=200, total_return_pct=1000)
    assert a < b
    c = PortfolioSelectionKey.from_metrics(max_flat_days=3, max_consecutive_losing_days=8, max_drawdown_pct=20, cagr_pct=40, total_return_pct=90)
    assert c < a
