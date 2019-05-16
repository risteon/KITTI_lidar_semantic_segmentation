#!/bin/bash
mkdir "$1" && sudo chown :85200 "$1" && sudo chmod 775 "$1" && sudo chmod g+s "$1"

