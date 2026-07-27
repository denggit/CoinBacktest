from analyze_tool.plugin_api import PluginRunResult, PriceHeatmapCell


def test_heatmap_serialization() -> None:
    result = PluginRunResult(
        markers=[],
        heatmap=[
            PriceHeatmapCell(
                start_timestamp="2026-06-01 00:00:00",
                end_timestamp="2026-06-01 00:05:00",
                price_low=1900,
                price_high=1905,
                intensity=0.8,
                side="long",
            )
        ],
    ).as_dict()
    assert result["heatmap"][0]["price_low"] == 1900
    assert result["heatmap"][0]["side"] == "long"
