#!/bin/bash
REPO_DIR="$(cd "$(dirname "$0")" && pwd)" 

caffeinate -s $REPO_DIR/publish.sh
