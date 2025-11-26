#!/usr/bin/bash
mkdir -p build
cmake build
cd build/ && make VERBOSE=1 -j 32 && ./run_all