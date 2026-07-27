# Liquidity Hunt Momentum R01 Fix 2

## Root cause

Pandas 2+ can preserve SQLite timestamps as `datetime64[us]`. The R01 fast paths
used `DatetimeIndex.view("int64")`, which returns integers in the index's native
unit. Those microsecond integers were compared with `Timestamp.value` and
`Timedelta.value`, which are nanoseconds. The 1000x unit mismatch caused
`searchsorted` to silently skip every daily Range-Bar slice, so the Books loader
was never called even though `.features.npz` files were valid.

## Fixes

1. All binary-search datetime axes are explicitly converted to `datetime64[ns]`
   before conversion to int64.
2. The fix covers:
   - daily Range-Bar slicing for Books loading;
   - Books `available_time` alignment;
   - signal-to-entry positioning;
   - exit positioning;
   - forward close/MFE/MAE path labels.
3. Books loading starts at `start_date`, not the Range-Bar warmup date. A trailing
   lead-in is still loaded for causal rolling references.
4. The research now fails fast when zero liquidity feature rows are loaded,
   instead of writing a misleading empty review pack.
5. Range Bars whose `end_ts` is later than the requested research end are removed.

## Validation

- R01 targeted tests: 18 passed.
- Loader/store/parser/causal/review-pack regression subset: 43 passed.
- Built-in self-test: passed.
- `compileall`: passed.
- New regression tests explicitly use `datetime64[us]` and reproduce the original
  silent Books-skip failure before the fix.

No raw Books or derived liquidity-map rebuild is required when existing day files
show `schema_version=2`, `book_feature_rows=86400`, and the loader returns rows.
