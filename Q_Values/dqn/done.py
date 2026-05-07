#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天标记一次骑手完成状态

跟踪每个骑手活跃的日期，当日期变化时标记骑手为完成。
确保每个骑手每天只被标记一次完成状态。
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DoneMarker:
    """
    每天标记一次骑手完成状态

    跟踪每个骑手最后活跃的日期，当当前日期与上次活跃日期不同时，
    标记该骑手为完成。确保每个骑手每天只被标记一次。
    """

    def __init__(self, idle_timeout_minutes: int = 30):
        """
        Args:
            idle_timeout_minutes: 保留参数，保持接口兼容，不再基于超时判断
        """
        self.idle_timeout_minutes = idle_timeout_minutes
        self._done: Dict[str, bool] = {}
        self._active_day: Dict[str, datetime] = {}

    def reset(self):
        """重置所有骑手的完成状态"""
        self._done.clear()
        self._active_day.clear()

    def reset_courier(self, cid: str):
        """重置指定骑手的完成状态"""
        self._done[cid] = False

    def is_done(self, cid: str) -> bool:
        """检查骑手是否已被标记为完成"""
        return self._done.get(cid, False)

    def update_activity(self, cid: str, current_time: datetime):
        """更新骑手活跃时间"""
        self._active_day[cid] = current_time
        self._done[cid] = False

    def check_done(self, courier, cid: str, current_time: datetime) -> bool:
        """
        检查骑手当天是否已完成

        当当前日期与骑手上次活跃日期不同时，标记为完成。
        确保每个骑手每天只被标记一次。

        Args:
            courier: 骑手状态字典（保留参数，保持接口兼容）
            cid: 骑手ID
            current_time: 当前仿真时间

        Returns:
            True 如果骑手刚被标记为完成
        """
        if cid not in self._active_day:
            self._active_day[cid] = current_time
            return False

        # 如果当前日期与上次活跃日期不同（过了一天），标记为完成
        if current_time.date() != self._active_day[cid].date():
            if not self._done.get(cid, False):
                self._done[cid] = True
                logger.debug(f"骑手 {cid} 已完成当天工作")
                return True

        return False
