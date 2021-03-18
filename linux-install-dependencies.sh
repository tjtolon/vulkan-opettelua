#!/bin/sh

sudo apt update

sudo apt-get install build-essential tar curl zip unzip

git clone https://github.com/microsoft/vcpkg

./vcpkg/boostrap_vcpkg.sh

./vcpkg/vcpkg --triplet=x64-linux install fmt spdlog