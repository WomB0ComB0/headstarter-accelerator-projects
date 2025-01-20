#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, create_engine
from typing import List
from datetime import datetime, timedelta
from typing import Generator
from app.models import MarketState, TradingSignal
from sqlalchemy import desc

from app.core.config import settings

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


router = APIRouter(prefix="/market", tags=["market"])


@router.get("/state/current", response_model=MarketState)
def get_current_market_state(session: Session = Depends(get_db)):
    """Get the most recent market state assessment"""
    statement = select(MarketState).order_by(desc(MarketState.timestamp)).limit(1)
    result = session.exec(statement).first()
    if not result:
        raise HTTPException(status_code=404, detail="No market state data available")
    return result


@router.get("/signals/active", response_model=List[TradingSignal])
def get_active_signals(session: Session = Depends(get_db), lookback_hours: int = 24):
    """Get active trading signals from the last 24 hours"""
    cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
    statement = select(TradingSignal).where(TradingSignal.timestamp >= cutoff_time)
    return session.exec(statement).all()


@router.get("/analysis/risk", response_model=dict)
def get_risk_analysis(session: Session = Depends(get_db)):
    """Get current risk analysis metrics"""
    state = session.exec(
        select(MarketState).order_by(desc(MarketState.timestamp)).limit(1)
    ).first()

    if not state:
        raise HTTPException(status_code=404, detail="No market state data available")

    return {
        "volatility": state.volatility,
        "value_at_risk": state.var_95,
        "max_drawdown": state.max_drawdown,
        "anomaly_probability": state.anomaly_score,
        "recommended_allocation": {
            "cash": state.recommended_cash_position,
            "equity": state.recommended_equity_position,
            "hedge": state.recommended_hedge_position,
        },
    }
