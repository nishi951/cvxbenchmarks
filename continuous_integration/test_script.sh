#!/usr/bin/env bash
if [[ $COVERAGE == "true" ]]; then
    py.test -cov .
else
    py.test
fi
