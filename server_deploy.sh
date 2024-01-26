#!/bin/bash
set -e

echo "Deploying application ..."

docker compose down

docker compose up --build -d