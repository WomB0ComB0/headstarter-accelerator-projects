#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter

router = APIRouter(prefix="/utils", tags=["utils"])


@router.get("/health-check/")
async def health_check() -> bool:
    return True
