#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    brew install armadillo
fi