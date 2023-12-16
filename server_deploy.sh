#!/bin/bash
set -e

echo "Deploying application ..."

git stash

git pull -X theirs

docker-compose down

docker-compose up --build -d