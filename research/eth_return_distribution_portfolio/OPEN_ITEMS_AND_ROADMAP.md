# Open Items and Roadmap

1. 本地运行 RDP-01，审核 2024 / 2025 / 2026 各 horizon 的 q50 Rank IC、decile spread、quantile coverage、pinball skill。
2. 若所有有方向 return horizon 都无稳定增量，停止堆普通 price/trade-bar 特征，直接进入 derivatives / books 信息源。
3. 若至少一个 2h/6h/24h horizon 稳定，优先实现 distribution -> target exposure，不再先发明交易规则。
4. RDP-02 数据源优先级：Funding+OI+Basis -> Liquidation -> Books -> Range Footprint -> Cross-exchange。
5. RDP-04 账户必须按 delta_notional 计手续费/滑点，并逐结算点计 funding。
6. 新未来数据到来前，任何历史结果都不能直接 live approve。
