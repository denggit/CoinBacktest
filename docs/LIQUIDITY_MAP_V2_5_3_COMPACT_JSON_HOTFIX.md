# Liquidity Map V2.5.3 — Compact JSON Hotfix

## Problem

V2.5.2 raised the visible-cell budget to 800,000 so shallow positive last-snapshot cells would not disappear. However, every heatmap cell was still serialized as a large JSON object containing repeated timestamps, labels, colors and a nested `fields` mapping.

For a multi-day request this could create a response of several hundred MB. The Python server or browser connection could be terminated while writing/reading the body, after which the frontend only reported:

```text
Unexpected end of JSON input
```

## Fix

- Historical bars still use the final completed Books snapshot.
- The current live bar still uses the latest Books snapshot.
- Wall logic and 24-hour causal color scaling are unchanged.
- Requests above 50,000 cells use compact columnar JSON instead of one object per cell.
- Repeated timestamps and source-snapshot metadata are stored once per chart column.
- Cell intensity is transmitted as an integer scaled by 10,000; ordering and displayed strength are preserved.
- Default render budget is reduced from 800,000 to 400,000, while the weak/medium/strong stratified reducer continues preserving shallow cells.
- JSON responses above 64 KiB are gzip-compressed when the browser supports gzip.
- Frontend error handling now distinguishes empty/truncated backend responses from normal plugin errors.
- Static `app.js` version is bumped so the browser does not keep the old decoder.

## Cache

Existing `period_end_v2` caches remain valid. No liquidity-map rebuild and no cache rebuild are required.

## Validation

```text
72 targeted tests passed
Analyze Tool selftest passed
JavaScript syntax check passed
```
