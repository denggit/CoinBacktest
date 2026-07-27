from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "analyze_tool" / "static"


def test_chart_toolbar_has_navigation_date_jump_color_range_and_audit() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "resetViewBtn",
        "autoPriceBtn",
        "goStartBtn",
        "goEndBtn",
        "jumpDate",
        "jumpDateBtn",
        "heatmapColorControls",
        "heatmapColorMin",
        "heatmapColorMax",
        "resetHeatmapColorBtn",
        "alignmentAuditBadge",
        "wallOverlayControl",
        "wallOverlayToggle",
        "wallOverlayLabel",
        "heatmapCellCard",
        "heatmapCellDetail",
    ):
        assert f'id="{element_id}"' in html


def test_period_end_heatmap_still_uses_exact_timestamp_coordinates() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function xForTimestampMs" in js
    assert "cell._startMs" in js
    assert "cell._endMs" in js
    heatmap_block = js.split("function ensureHeatmapLayer", 1)[1].split("function heatmapAtIndex", 1)[0]
    assert "timeSpan" in heatmap_block
    assert "effectiveHeatmapIntensity(cell)" in heatmap_block


def test_heatmap_cell_hover_uses_exact_time_and_price_bounds() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function heatmapCellsAtPoint" in js
    assert "timestampMs >= cell._startMs && timestampMs < cell._endMs" in js
    assert "price >= low && price < high" in js
    assert "function renderHeatmapCellDetail" in js
    assert "source_snapshot_end" in js
    assert "rolling_large_threshold" in js


def test_price_axis_scaling_is_anchored_and_shift_enables_pan_xy() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "state.priceDragAnchorPrice = priceForY" in js
    assert "anchor - (anchor - startMin) * factor" in js
    assert "anchor + (startMax - anchor) * factor" in js
    assert "state.dragMode = overPriceAxis ? (e.shiftKey ? 'price-pan' : 'price-scale') : (e.shiftKey ? 'pan-xy' : 'pan-x')" in js
    assert "state.dragMode === 'pan-xy' && Math.abs(dy) > 1" in js


def test_heatmap_matrix_is_pixelated_cached_and_color_range_invalidates_cache() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "ctx.imageSmoothingEnabled = false" in js
    assert "lctx.imageSmoothingEnabled = false" in js
    assert "state.heatmapLayerKey" in js
    assert "liquidityHeatColor" in js
    assert "state.heatmapColorMinPct" in js
    assert "state.heatmapColorMaxPct" in js
    assert "function setHeatmapColorRange" in js


def test_chart_wall_clock_time_is_timezone_stable() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "const chartTime = parseTimestampMs(candle.timestamp)" in js
    assert "Date.UTC(" in js
    assert "getUTCFullYear" in js
    assert "const target = parseTimestampMs(timestamp)" in js
    assert "source_time: candle.time, time: chartTime" in js


def test_persistent_wall_overlay_is_optional_and_drawn_as_fixed_deep_blue_rectangles() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "state.showWallOverlay" in js
    assert "data.price_regions" in js
    assert "function buildPriceRegionChains" in js
    assert "function drawWallRegionChain" in js
    assert "function drawPriceRegions" in js
    assert "previousChain.regions.push(region)" in js
    assert "rectangle_price_low" in js
    assert "rectangle_price_high" in js
    assert "ctx.strokeRect" in js
    assert "#00AEEF" in js
    assert "wall_overlay_default" in js


def test_wall_overlay_does_not_draw_stepped_or_dashed_lifecycle_shapes() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    segment = js.split("function drawWallRegionChain", 1)[1].split("function drawPriceRegions", 1)[0]
    assert "ctx.strokeRect" in segment
    assert "ctx.lineTo" not in segment
    assert "setLineDash([])" in segment
    assert "[7, 5]" not in segment
    assert "[2, 4]" not in segment


def test_coinglass_palette_has_clear_ivory_salmon_and_dark_magenta_extremes() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    palette = js.split("function liquidityHeatColor", 1)[1].split("function heatmapLayerKey", 1)[0]
    assert "r: 255, g: 254, b: 246" in palette
    assert "r: 223, g: 130, b: 129" in palette
    assert "r: 119, g: 45, b: 109" in palette
    assert "const alpha = 0.90 + 0.08 * confidence" in js
    assert "r: 251, g: 191, b: 36" not in palette


def test_coinglass_light_canvas_is_used_for_chart_and_shell() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert "#fffefa" in js
    assert "background: #f4f6f8" in css
    assert "background: #fffefa" in css
    assert "color-scheme: light" in css


def test_heatmap_upper_control_is_a_saturation_threshold_with_50pct_default() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="heatmapColorMax" type="range" min="5" max="100" step="1" value="50"' in html
    assert "隐藏低于" in html
    assert "最深达到" in html
    assert "clamp(Number(maxPct), 5, 100)" in js
    assert "Math.min(100, min + 1)" in js
    assert "heatmap_color_max_pct ?? 50" in js


def test_wall_overlay_rectangles_do_not_draw_labels_over_the_chart() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    segment = js.split("function drawWallRegionChain", 1)[1].split("function drawPriceRegions", 1)[0]
    assert "fillText" not in segment
