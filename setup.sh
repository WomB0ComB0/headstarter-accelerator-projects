#!/bin/bash

if [-d venv]; then
    echo "venv exists"
else
    echo "venv does not exist"
    python3 -m venv venv
fi