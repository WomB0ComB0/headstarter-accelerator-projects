"""
  @file: api/config.py
  @author: Mike Odnis
  @date: 2024-12-18
  @description: Configuration for the API
"""

import os
from typing import Optional, Dict, TypedDict

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    def __init__(self):
        self.vars = self.get_var()

    def get_var(self) -> Optional[Dict[str, str]]:
        with open(os.path.join(basedir, ".env"), "r", encoding="utf-8") as f:
            return {line.split("=")[0]: line.split("=")[1] for line in f.readlines()}
