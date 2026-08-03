# ETH AI Trading R02：三-Sleeve框架运行说明

## 目的

R02不训练模型，只固定Short-horizon、Intraday trend、Swing三类市场过程的接口边界，防止后续再次把几分钟与几天的标签混入一个模型。

## 运行

```bat
python research\eth_ai_trading\02_upgrade_three_sleeve_framework.py
```

输出：

```text
data\reports\research\eth_ai_trading\02_three_sleeve_framework
```

## 核心合同

- `src.ai_research.sleeves.SleeveSpec`
- `src.ai_research.sleeves.ModelEvidence`
- `src.ai_research.sleeves.TradeCandidate`
- `src.ai_research.sleeves.TargetPositionDecision`

所有对象均为纯计算/数据合同，不访问数据库、交易所或AetherEdge。
