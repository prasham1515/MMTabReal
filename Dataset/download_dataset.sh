#!/usr/bin/env bash

git clone https://huggingface.co/datasets/MMTBench/MMTabReal

cd MMTabReal

rm -rf .git
rm -f all.zip

curl -L -o all.zip https://huggingface.co/datasets/MMTBench/MMTabReal/resolve/main/all.zip

unzip all.zip