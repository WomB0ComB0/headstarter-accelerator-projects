#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class MarketState(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    anomaly_score: float
    is_anomaly: bool
    confidence: float = Field(default=0.0)

    # Portfolio adjustments
    recommended_cash_position: float
    recommended_equity_position: float
    recommended_hedge_position: float

    # Risk metrics
    volatility: float
    var_95: float  # Value at Risk at 95% confidence
    max_drawdown: float


class TradingSignal(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signal_type: str  # 'BUY', 'SELL', 'HEDGE'
    strength: float  # Signal strength from 0 to 1
    asset_class: str  # 'EQUITY', 'BONDS', 'COMMODITIES', etc.
    description: Optional[str] = None
