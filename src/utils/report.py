#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import pandas as pd

from .log import get_logger

logger = get_logger(__name__)


def _safe_report_name(strategy_name: str) -> str:
    return str(strategy_name).replace(' ', '_').replace('/', '_').replace(':', '')


def _create_text_report_handler(txt_filename: str) -> logging.FileHandler:
    """Create a plain-text report handler.

    The txt file is for reading the report content, not for debugging logs,
    so it intentionally excludes timestamp/module/level metadata.
    Console logging keeps its original formatter.
    """
    handler = logging.FileHandler(txt_filename, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def print_full_report(trade_history, df, initial_capital, capital, strategy_name, total_days, ai_enabled, symbol=None, report_dir=None):
    """
    1:1 还原用户最喜爱的深度量化报告格式，包含所有量化维度与详细时间戳。

    如果传入 report_dir，会额外把本函数 logger.info 打出的完整报告保存为 txt。
    txt 文件名和本函数导出的交易明细 csv 使用同一套命名：
        {safe_name}_{start_str}_{end_str}_{ai_enabled}.txt
        {safe_name}_{start_str}_{end_str}_{ai_enabled}.csv
    """
    # --- 0. 报告目录与文件名预处理：必须放在第一条 logger.info 之前，才能完整捕获日志 ---
    if symbol is None:
        symbol = "unknown"
    safe_symbol = symbol.replace('-', '_')

    current_file = os.path.abspath(__file__)
    # 向上推三层：report.py -> utils -> src -> 根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    data_reports_dir = os.path.join(project_root, 'data', 'reports')
    report_dir = report_dir if report_dir else os.path.join(data_reports_dir, strategy_name.split(" ")[0])
    os.makedirs(report_dir, exist_ok=True)

    safe_name = _safe_report_name(strategy_name)

    if df is not None and len(df.index) > 0:
        start_str = df.index[0].strftime('%Y-%m-%d')
        end_str = df.index[-1].strftime('%Y-%m-%d')
    else:
        start_str = "unknown_start"
        end_str = "unknown_end"

    csv_filename = os.path.join(report_dir, f"{safe_name}_{start_str}_{end_str}_{ai_enabled}.csv")
    txt_filename = os.path.join(report_dir, f"{safe_name}_{start_str}_{end_str}_{ai_enabled}.txt")

    text_handler = _create_text_report_handler(txt_filename)
    logger.addHandler(text_handler)

    try:
        total_trades = len(trade_history)
        if total_trades == 0:
            logger.info(f"\n=== 🚀 {strategy_name} ===")
            logger.info("未产生任何交易，请检查策略逻辑或信号生成。")
            logger.info(f"\n📄 完整报告日志已导出至: {os.path.abspath(txt_filename)}")
            return

        # --- 1. 核心数据预计算 ---
        win_trades = sum(1 for t in trade_history if t['pnl'] > 0)
        gross_profit = sum(t['pnl'] for t in trade_history if t['pnl'] > 0)
        gross_loss = sum(abs(t['pnl']) for t in trade_history if t['pnl'] <= 0)
        total_fees_paid = sum(t['fee'] for t in trade_history)

        capital_curve = [initial_capital]
        peak_capital = initial_capital
        max_drawdown_pct = 0.0
        trade_returns = []

        for t in trade_history:
            pnl = t['pnl']
            cap_before = t['capital'] - pnl
            t['return_pct'] = pnl / cap_before if cap_before > 0 else 0
            trade_returns.append(t['return_pct'])
            if t['capital'] > peak_capital:
                peak_capital = t['capital']
            dd = (peak_capital - t['capital']) / peak_capital if peak_capital > 0 else 0
            max_drawdown_pct = max(max_drawdown_pct, dd)

        win_returns = [t['return_pct'] for t in trade_history if t['return_pct'] > 0]
        loss_returns = [t['return_pct'] for t in trade_history if t['return_pct'] < 0]

        win_rate = win_trades / total_trades
        loss_rate = 1 - win_rate

        avg_win_pct = sum(win_returns) / len(win_returns) if win_returns else 0
        avg_loss_pct = sum(abs(r) for r in loss_returns) / len(loss_returns) if loss_returns else 0

        pnl_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else float('inf')

        gross_win_pct = sum(win_returns)
        gross_loss_pct = sum(abs(r) for r in loss_returns)
        profit_factor = gross_win_pct / gross_loss_pct if gross_loss_pct > 0 else float('inf')

        expectancy_pct = (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct)

        years = total_days / 365.25
        cagr = ((capital / initial_capital) ** (1 / years) - 1) if years > 0 and capital > 0 else 0
        calmar = cagr / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')

        # 计算夏普比率
        if len(trade_returns) > 1:
            std_dev = np.std(trade_returns)
            sharpe = (np.mean(trade_returns) / std_dev) * np.sqrt(total_trades * (1 / years)) if std_dev > 0 else 0
        else:
            sharpe = 0

        # --- 2. 开始打印头部 ---
        logger.info(f"\n=== 🚀 启动 {strategy_name} | {start_str} 至 {end_str} ({total_days:.1f} 天) ===")

        logger.info("\n" + "=" * 65)
        logger.info(f" 📊 {strategy_name} - 深度量化绩效报告")
        logger.info("=" * 65)

        # --- 3. 逐年绩效拆解 ---
        logger.info("\n" + "=" * 65)
        logger.info(" 📅 逐年绩效拆解 (Yearly Breakdown)")
        logger.info("=" * 65)
        logger.info(f"{'年份':<6} | {'初始资金':<10} | {'净盈亏':<10} | {'当年收益率':<10} | {'胜率':<6} | {'最大回撤':<8}")
        logger.info("-" * 65)

        current_year_cap = initial_capital
        for y in sorted(df.index.year.unique()):
            trades_y = [t for t in trade_history if t['exit_time'].year == y]
            if not trades_y:
                continue
            y_wins = sum(1 for t in trades_y if t['pnl'] > 0)
            y_pnl = sum(t['pnl'] for t in trades_y)
            y_roi = y_pnl / current_year_cap

            y_peak, y_max_dd, temp_c = current_year_cap, 0.0, current_year_cap
            for t in trades_y:
                temp_c += t['pnl']
                y_peak = max(y_peak, temp_c)
                dd = (y_peak - temp_c) / y_peak if y_peak > 0 else 0
                y_max_dd = max(y_max_dd, dd)

            logger.info(
                f"{y:<6} | ${current_year_cap:<9.2f} | ${y_pnl:<+9.2f} | {y_roi * 100:>+8.2f}%  | {y_wins / len(trades_y) * 100:>5.1f}% | {-y_max_dd * 100:>6.2f}%")
            current_year_cap += y_pnl

        # --- 4. 核心量化指标 ---
        logger.info("\n" + "-" * 65)
        logger.info(" 📈 核心量化指标 (Core Metrics)")
        logger.info("-" * 65)
        logger.info(f"测试跨度 (Duration):      {total_days:.1f} 天 ({years:.2f} 年)")
        logger.info(f"总交易次数 (Total Trades):  {total_trades}")
        logger.info(f"胜率 (Win Rate):          {win_rate * 100:.2f}%")
        logger.info(f"平均净盈利 (Avg Win):     +{avg_win_pct * 100:.2f}%")
        logger.info(f"平均净亏损 (Avg Loss):    -{avg_loss_pct * 100:.2f}%")
        logger.info(f"净盈亏比 (PnL Ratio):     {pnl_ratio:.2f}")
        logger.info(f"盈利因子 (Profit Factor): {profit_factor:.2f}")

        sign = "+" if expectancy_pct > 0 else ""
        logger.info(f"单笔期望值 (Expectancy):  {sign}{expectancy_pct * 100:.2f}%")

        # --- 5. 风险与财务评估 ---
        logger.info("\n" + "-" * 65)
        logger.info(" 🛡️ 风险与财务评估 (Risk & Finance)")
        logger.info("-" * 65)
        logger.info(f"最大回撤 (Max Drawdown):  {max_drawdown_pct * 100:.2f}%")
        logger.info(f"夏普比率 (Sharpe Ratio):  {sharpe:.2f}")
        logger.info(f"卡玛比率 (Calmar Ratio):  {calmar:.2f}")
        logger.info(f"给交易所交的手续费总计:   -${total_fees_paid:.2f}")

        # --- 6. Top 5 史诗级战绩 ---
        for t in trade_history:
            dur = t['exit_time'] - t['entry_time']
            h = int(dur.total_seconds() // 3600)
            t['duration_str'] = f"{h // 24}天 {h % 24}小时" if h >= 24 else f"{h}小时"

        wins_top = sorted([t for t in trade_history if t['return_pct'] > 0], key=lambda x: x['return_pct'], reverse=True)[:5]
        loss_top = sorted([t for t in trade_history if t['return_pct'] < 0], key=lambda x: x['return_pct'])[:5]

        logger.info("\n" + "🏆" * 3 + " 盈利 Top 5 史诗级交易 " + "🏆" * 3)
        logger.info("-" * 65)
        for i, t in enumerate(wins_top):
            logger.info(
                f"{i + 1}. [{t['type']}] 进: {t['entry_time'].strftime('%Y-%m-%d %H:%M')} | 出: {t['exit_time'].strftime('%Y-%m-%d %H:%M')} | 历时: {t['duration_str']} | 净赚: +{t['return_pct'] * 100:.2f}%")

        logger.info("\n" + "🩸" * 3 + " 亏损 Top 5 极度考验 " + "🩸" * 3)
        logger.info("-" * 65)
        for i, t in enumerate(loss_top):
            logger.info(
                f"{i + 1}. [{t['type']}] 进: {t['entry_time'].strftime('%Y-%m-%d %H:%M')} | 出: {t['exit_time'].strftime('%Y-%m-%d %H:%M')} | 历时: {t['duration_str']} | 净亏: {t['return_pct'] * 100:.2f}%")

        # --- 7. 最终结算 ---
        logger.info("\n" + "=" * 65)
        logger.info(f"初始资金 (Initial Cap):   ${initial_capital:.2f}")
        logger.info(f"最终资金 (Final Cap):     ${capital:.2f}")
        logger.info(
            f"总净利润 (Net PnL):       +${(capital - initial_capital):.2f} (总收益率: {(capital - initial_capital) / initial_capital * 100:.2f}%)")
        logger.info(f"复合年化收益率 (CAGR):    {cagr * 100:.2f}%")
        logger.info("=" * 65)

        # --- 8. CSV 导出 ---
        export_df = pd.DataFrame(trade_history)
        export_df.rename(columns={
            'entry_time': 'Entry_Time', 'exit_time': 'Exit_Time', 'type': 'Type',
            'entry': 'Entry_Price', 'exit': 'Exit_Price', 'pnl': 'Net_PnL',
            'fee': 'Fee', 'capital': 'Capital', 'mfe_r': 'MFE(R)', 'mae_r': 'MAE(R)'
        }, inplace=True)

        export_df.to_csv(csv_filename, index=False)
        logger.info(f"\n📂 交易明细已导出至: {os.path.abspath(csv_filename)}")
        logger.info(f"📄 完整报告日志已导出至: {os.path.abspath(txt_filename)}")

    finally:
        logger.removeHandler(text_handler)
        text_handler.close()
