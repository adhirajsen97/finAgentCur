#!/bin/bash

# FinAgent Enhanced API - Docker Test Script
# This script tests the Docker build and deployment locally

set -e

echo "🐳 FinAgent Enhanced API - Docker Test"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✅ Docker is installed${NC}"

# Build the Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -t finagent-enhanced:latest . --no-cache

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Test the Docker image
echo -e "${YELLOW}🚀 Starting container for testing...${NC}"
docker run -d --name finagent-test -p 8000:8000 -e PORT=8000 finagent-enhanced:latest

# Wait for container to start
echo -e "${YELLOW}⏳ Waiting for container to start...${NC}"
sleep 10

# Test health endpoint
echo -e "${YELLOW}🏥 Testing health endpoint...${NC}"
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    docker logs finagent-test
    docker stop finagent-test
    docker rm finagent-test
    exit 1
fi

# Test workflow endpoint
echo -e "${YELLOW}📋 Testing workflow endpoint...${NC}"
if curl -f http://localhost:8000/api/workflow-guide > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Workflow endpoint working${NC}"
else
    echo -e "${RED}❌ Workflow endpoint failed${NC}"
fi

# Test root endpoint
echo -e "${YELLOW}🏠 Testing root endpoint...${NC}"
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Root endpoint working${NC}"
else
    echo -e "${RED}❌ Root endpoint failed${NC}"
fi

# Show container logs
echo -e "${YELLOW}📝 Container logs:${NC}"
docker logs finagent-test --tail 20

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
docker stop finagent-test
docker rm finagent-test

echo -e "${GREEN}🎉 Docker test completed successfully!${NC}"
echo ""
echo "To run the container manually:"
echo "docker run -p 8000:8000 -e PORT=8000 finagent-enhanced:latest"
echo ""
echo "To access the API:"
echo "- Health: http://localhost:8000/health"
echo "- Workflow: http://localhost:8000/workflow"
echo "- Docs: http://localhost:8000/docs" 