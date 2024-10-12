#!/bin/bash

for i in {1..14} ; do
    mkdir -p "week-$i"
    touch "week-$i/.gitkeep"
done