#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter
from app.api.routes import utils, market

api_router = APIRouter()
api_router.include_router(utils.router)
api_router.include_router(market.router)
