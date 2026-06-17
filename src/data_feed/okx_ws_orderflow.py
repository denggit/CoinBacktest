import argparse
import asyncio
import csv  # 新增 csv 模块
import datetime
import json
import logging
import os
import signal
import sys
import time
from collections import deque

import websockets

# 添加项目根目录到 Python 路径
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.log import get_logger
from src.utils.email_sender import send_trading_signal_email

logger = get_logger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class OrderFlowSniper:
    def __init__(self, symbol="ETH-USDT-SWAP", mode="collect"):
        self.symbol = symbol
        self.mode = mode  # 'collect' 或 'live'
        # 使用 AWS 专线域名，在东京节点极其稳定
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"

        # 实时状态
        self.cvd = 0.0
        self.current_price = 0.0

        # 内存降维：使用双端队列存储过去 300 个“10秒快照” (回看过去 50 分钟)
        self.snapshots = deque(maxlen=300)
        self.last_snapshot_time = time.time()
        self.last_heartbeat = time.time()

        # 邮件发送频率控制 (至少间隔10分钟，避免重复报警)
        self._last_email_sent_time = 0
        self._email_cooldown = 600  # 10分钟，单位秒

        # ==========================================
        # 📊 数据科考船：CSV 归档配置
        # ==========================================
        self.active_trackings = []
        self.csv_file = os.path.join(project_root, "data", "bounce_records.csv")

        # 🌟 新增：独立的时间锁与价格记忆
        self.last_broad_trigger_time = 0
        self.last_broad_trigger_price = 0.0  # 新增：记忆宽口径触发时的价格

        self.last_strict_trigger_time = 0
        self.last_strict_trigger_price = 0.0  # 新增：记忆严口径触发时的价格

        # 确保目录存在，并初始化 CSV 表头
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['触发时间', 'CVD砸盘量(万刀)', 'CVD反转量(万刀)', '偏离前低(刀)', '触发价格', '反弹最高价',
                     '最大反弹幅度(%)', '追踪耗时(秒)', '结束原因'])

    async def connect_and_listen(self):
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": self.symbol}]
        }

        while True:
            try:
                logger.info(f"🚀 正在连接 OKX 订单流极速通道 ({self.symbol})...")
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("✅ 接入成功！[双轨制雷达] 运行中：宽口径存CSV，严口径发邮件...")

                    while True:
                        response = await ws.recv()
                        data = json.loads(response)

                        if 'data' in data:
                            self._process_ticks(data['data'])

            except Exception as e:
                logger.error(f"❌ 链路断开，准备重连: {e}")
                await asyncio.sleep(3)

    def _process_ticks(self, trades):
        current_ts = time.time()

        for trade in trades:
            self.current_price = float(trade['px'])
            size = float(trade['sz'])
            side = trade['side']

            # CVD 核心累计 (买入增加，卖出减少)
            if side == 'buy':
                self.cvd += size
            else:
                self.cvd -= size

        # 1. 科考船功能：动态更新反弹高点并归档 CSV
        self._update_trackings(current_ts)

        # 2. 触发快照与背离检测机制 (每 10 秒拍一次照，绝不刷屏)
        if current_ts - self.last_snapshot_time >= 10:
            self._take_snapshot(current_ts)
            self._detect_absorption_divergence()
            self.last_snapshot_time = current_ts

        # 3. 心跳日志 (每 1 分钟报备一次，让你知道它没死机)
        if current_ts - self.last_heartbeat >= 60:
            logger.debug(
                f"💓 [雷达扫掠中] 现价: {self.current_price} | 正在追踪的底层暗流: {len(self.active_trackings)} 个")
            self.last_heartbeat = current_ts

    def _update_trackings(self, current_ts):
        """🌟 动态更新反弹高点，并判定是否结束追踪写入 CSV"""
        for track in self.active_trackings[:]:
            # 刷新反弹最高点
            if self.current_price > track['max_price']:
                track['max_price'] = self.current_price

            end_reason = None
            # 结束条件 1: 价格跌破了触发时的最低防线 (比如跌破触发价 3 刀，设宽一点防止被假跌破洗掉)
            if self.current_price < (track['entry_price'] - 3.0):
                end_reason = "破位止损"
            # 结束条件 2: 追踪时间满 15 分钟
            elif current_ts - track['entry_time'] > 900:
                end_reason = "时间到了(15分钟)"

            if end_reason:
                bounce_pct = (track['max_price'] - track['entry_price']) / track['entry_price'] * 100
                duration = current_ts - track['entry_time']

                try:
                    with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.datetime.fromtimestamp(track['entry_time']).strftime('%Y-%m-%d %H:%M:%S'),
                            round(track['cvd_delta_usdt'] / 10000, 2),
                            round(track['micro_cvd_delta_usdt'] / 10000, 2),  # 🌟 新增：写入反转量(万刀)
                            round(track['price_diff_pct'], 2),
                            track['entry_price'],
                            track['max_price'],
                            round(bounce_pct, 4),
                            round(duration, 1),
                            end_reason
                        ])
                    logger.info(
                        f"📊 记录归档 [{end_reason}] -> CVD砸盘: {track['cvd_delta_usdt'] / 10000:.1f}万 | 反弹幅度: {bounce_pct:.3f}% | 已写入CSV")
                except Exception as e:
                    logger.error(f"CSV写入失败: {e}")

                self.active_trackings.remove(track)

    def _take_snapshot(self, ts):
        """将当前的价格和 CVD 压入历史窗口"""
        self.snapshots.append({
            'ts': ts,
            'price': self.current_price,
            'cvd': self.cvd
        })

    def _detect_absorption_divergence(self):
        """🌟 双轨制核心武器：非破坏性独立时间锁 + 局部前低修复"""

        # ==========================================
        # 🐛 Bug修复 1：冷启动防御
        # ==========================================
        # 强制系统至少收集 15 分钟（90个快照）的数据后，雷达才允许开机！
        if len(self.snapshots) < 90:
            return

        # ==========================================
        # 🐛 Bug修复 2：切除“历史阴影”，只找波段低点
        # ==========================================
        LOOKBACK_WINDOW = 90  # 只回看过去 15 分钟
        past_snapshots = list(self.snapshots)[-LOOKBACK_WINDOW:-1]
        lowest_snap = min(past_snapshots, key=lambda x: x['price'])
        current_snap = self.snapshots[-1]

        # 🌟 升级为跨币种通用的百分比算法：
        price_diff_pct = (current_snap['price'] - lowest_snap['price']) / lowest_snap['price'] * 100

        RECENT_WINDOW = 18  # 180秒
        snapshot_3min_ago = self.snapshots[-RECENT_WINDOW]
        recent_cvd_delta_contracts = current_snap['cvd'] - snapshot_3min_ago['cvd']
        CONTRACT_SIZE = 0.1
        recent_cvd_delta_usdt = recent_cvd_delta_contracts * CONTRACT_SIZE * current_snap['price']

        last_snap = self.snapshots[-2]
        micro_cvd_delta_contracts = current_snap['cvd'] - last_snap['cvd']
        micro_cvd_delta_usdt = micro_cvd_delta_contracts * CONTRACT_SIZE * current_snap['price']

        # 距离上一次前低点至少过去了 20 秒
        time_passed = (current_snap['ts'] - lowest_snap['ts']) > 20

        # ==========================================
        # 🧪 外层网：科考船宽口径 (时间锁：300秒内不重复建档)
        # ==========================================
        # 🌟 调整：上限放宽至 +5.0，容纳长下影线
        broad_price_ok = -3.0 <= price_diff_pct <= 5.0
        broad_cvd_ok = recent_cvd_delta_usdt < -2_000_000
        broad_turn_ok = micro_cvd_delta_usdt > 50_000

        # 🌟 智能冷却锁：过了5分钟，或者价格比上次报警低了至少 2 刀（二次探底）
        broad_time_ok = (current_snap['ts'] - self.last_broad_trigger_time) > 300
        broad_price_override = current_snap['price'] < (self.last_broad_trigger_price - 2.0)
        broad_cooldown_ok = broad_time_ok or broad_price_override

        if broad_price_ok and broad_cvd_ok and broad_turn_ok and time_passed and broad_cooldown_ok:
            logger.warning(
                f"🎯 捕获暗流信号！砸盘: ${abs(recent_cvd_delta_usdt) / 10000:.1f}万，偏离前低: {price_diff_pct:.2f}刀，CVD反转：{micro_cvd_delta_usdt / 10000:.2f}万。加入CSV追踪队列...")

            self.active_trackings.append({
                'entry_time': current_snap['ts'],
                'entry_price': current_snap['price'],
                'cvd_delta_usdt': recent_cvd_delta_usdt,
                'micro_cvd_delta_usdt': micro_cvd_delta_usdt,  # 🌟 新增：把反转量塞进字典
                'price_diff_pct': price_diff_pct,
                'max_price': current_snap['price']
            })

            # 更新时间和价格记忆
            self.last_broad_trigger_time = current_snap['ts']
            self.last_broad_trigger_price = current_snap['price']

        # ==========================================
        # ⚔️ 内层网：实盘严口径 (独立时间锁：300秒内不重复发邮件)
        # ==========================================
        # 🌟 调整：上限放宽至 +4.0，绝不放过极速拉升的黄金坑
        strict_price_ok = -2.0 <= price_diff_pct <= 4.0
        strict_cvd_ok = recent_cvd_delta_usdt < -5_000_000
        strict_turn_ok = micro_cvd_delta_usdt > 150_000

        # 🌟 智能冷却锁：过了5分钟，或者价格比上次报警低了至少 3 刀（代表散户止损被真实击穿）
        strict_time_ok = (current_snap['ts'] - self.last_strict_trigger_time) > 300
        strict_price_override = current_snap['price'] < (self.last_strict_trigger_price - 3.0)
        strict_cooldown_ok = strict_time_ok or strict_price_override

        if strict_price_ok and strict_cvd_ok and strict_turn_ok and time_passed and strict_cooldown_ok:
            logger.warning("\n" + "🟢" * 25)
            logger.warning(f"🚨 [流速级抄底绝杀] 发现深海冰山！散户正在被集中血洗！")
            logger.warning(
                f"💥 爆量数据: 就在刚刚的 【3分钟】 内，市场瞬间涌入了 ${abs(recent_cvd_delta_usdt) / 10000:.1f} 万美金的市价砸盘！")
            logger.warning(f"偏离前低: {price_diff_pct:.2f}刀，CVD反转：{micro_cvd_delta_usdt / 10000:.2f}万")
            logger.warning(f"🛡️ 盘口真相: 价格被死死托在 {current_snap['price']} 附近，根本跌不下去。")
            logger.warning("🎯 战术结论: 典型的抛售高潮 (Selling Climax) + 机构限价吸收！准备抢反弹！")
            logger.warning("🟢" * 25 + "\n")

            asyncio.create_task(self._send_bottom_fishing_email(
                symbol=self.symbol,
                price=current_snap['price'],
                cvd_delta_usdt=recent_cvd_delta_usdt,
                time_window_minutes=3.0
            ))

            # 更新时间和价格记忆
            self.last_strict_trigger_time = current_snap['ts']
            self.last_strict_trigger_price = current_snap['price']

    async def _send_bottom_fishing_email(self, symbol: str, price: float,
                                         cvd_delta_usdt: float, time_window_minutes: float):
        """发送抄底机会邮件通知"""
        # collect模式下不发送交易信号邮件
        if self.mode == "collect":
            logger.info(f"📝 [收集模式] 检测到抄底信号但不发送邮件: 价格={price}, 砸盘=${abs(cvd_delta_usdt):,.0f} USDT")
            return

        current_time = time.time()
        if current_time - self._last_email_sent_time < self._email_cooldown:
            logger.debug(
                f"邮件发送频率限制，跳过本次发送。还需等待 {self._email_cooldown - (current_time - self._last_email_sent_time):.0f} 秒")
            return

        try:
            signal_type = "流速级抄底绝杀 (极端V反包容版)"
            detection_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            details = f"""
🚨 检测到机构恐慌吸收信号！

📊 交易对: {symbol}
🕐 检测时间: {detection_time}
💰 当前价格: {price:.2f}
📉 3分钟内净卖出: ${abs(cvd_delta_usdt):,.0f} USDT
⏱️ 时间窗口: {time_window_minutes} 分钟
📈 信号类型: {signal_type}

💡 核心逻辑:
1. 价格被压制在近期低点附近，或已完成极限插针V反！
2. 散户在短时间内疯狂抛售 {abs(cvd_delta_usdt) / 10000:.1f} 万美金
3. 价格未能继续下跌，并且主力已拍出 >5 万美金的市价单反击！

🎯 操作建议: 准备抢反弹，关注短期反转机会 (建议止损设置在 -4.0 刀)
            """

            success = await send_trading_signal_email(
                symbol=symbol,
                signal_type=signal_type,
                price=price,
                details=details
            )

            if success:
                self._last_email_sent_time = time.time()
                logger.info("✅ 抄底机会邮件发送成功")
            else:
                logger.warning("⚠️ 抄底机会邮件发送失败")

        except Exception as e:
            logger.error(f"发送抄底机会邮件时发生错误: {e}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="订单流狙击手 (OrderFlowSniper)")
    parser.add_argument('--symbol', type=str, default='ETH-USDT-SWAP',
                        help='交易对符号，例如: ETH-USDT-SWAP, BTC-USDT-SWAP')
    parser.add_argument('--mode', type=str, default='collect',
                        choices=['collect', 'live'],
                        help="运行模式: 'collect' (只收集数据不发邮件) 或 'live' (发送邮件通知)")
    args = parser.parse_args()


    # 🌟 新增：信号转换器。当收到 kill -15 (SIGTERM) 时，主动抛出 KeyboardInterrupt
    def handle_sigterm(*args):
        logger.warning("🔔 收到 kill -15 (SIGTERM) 信号！转换为安全迫降指令...")
        raise KeyboardInterrupt()


    # 监听 kill -15 信号
    signal.signal(signal.SIGTERM, handle_sigterm)

    sniper = OrderFlowSniper(symbol=args.symbol, mode=args.mode)
    logger.info(f"🚀 订单流狙击手启动: 交易对={args.symbol}, 模式={args.mode}")

    try:
        asyncio.run(sniper.connect_and_listen())
    except KeyboardInterrupt:
        logger.info("\n⚠️ 准备执行安全迫降，保存内存数据...")

        # 强行结算还没追踪完的订单
        if sniper.active_trackings:
            current_ts = time.time()
            for track in sniper.active_trackings:
                bounce_pct = (track['max_price'] - track['entry_price']) / track['entry_price'] * 100
                duration = current_ts - track['entry_time']

                try:
                    with open(sniper.csv_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.datetime.fromtimestamp(track['entry_time']).strftime('%Y-%m-%d %H:%M:%S'),
                            round(track['cvd_delta_usdt'] / 10000, 2),
                            round(track['micro_cvd_delta_usdt'] / 10000, 2),  # 🌟 新增：强行结算时也写入
                            round(track['price_diff_pct'], 2),
                            track['entry_price'],
                            track['max_price'],
                            round(bounce_pct, 4),
                            round(duration, 1),
                            "程序重启中断(强制结算)"
                        ])
                except Exception as e:
                    pass
            logger.info(f"✅ 完美！已将 {len(sniper.active_trackings)} 个未完成的追踪记录抢救至 CSV！")

        logger.info("⏹️ 订单流狙击手已安全撤离。可以放心重启了！")
