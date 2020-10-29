#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /mnt/models/config/config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
