#!/bin/bash
# Run 500k Conversation Scale Test
# This script helps run the scale test with different configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo -e "${CYAN}  500K CONVERSATION SCALE TEST RUNNER${NC}"
echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo -e "${YELLOW}Please create a virtual environment first:${NC}"
    echo "  uv venv --python 3.12"
    echo "  source .venv/bin/activate"
    echo "  uv pip install -r requirements.txt"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Please create .env file from .env.example:${NC}"
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your API keys"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Parse command line arguments
TEST_TYPE=${1:-full}

case $TEST_TYPE in
    "quick")
        echo -e "${CYAN}Running quick test (10k conversations)...${NC}"
        python tests/test_scale_500k.py --quick-test
        ;;

    "retrieval-only")
        echo -e "${CYAN}Running retrieval tests only (skipping population)...${NC}"
        python tests/test_scale_500k.py --skip-population
        ;;

    "full")
        echo -e "${YELLOW}⚠ WARNING: This will populate database with 500k conversations${NC}"
        echo -e "${YELLOW}   This process may take several hours.${NC}"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python tests/test_scale_500k.py --target 500000
        else
            echo -e "${RED}Test cancelled${NC}"
            exit 0
        fi
        ;;

    "custom")
        TARGET=${2:-100000}
        echo -e "${CYAN}Running custom test with ${TARGET} conversations...${NC}"
        python tests/test_scale_500k.py --target $TARGET
        ;;

    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo ""
        echo "Usage: $0 [test_type] [options]"
        echo ""
        echo "Test types:"
        echo "  quick          - Quick test with 10k conversations"
        echo "  retrieval-only - Run retrieval tests only (skip population)"
        echo "  full           - Full test with 500k conversations (default)"
        echo "  custom [N]     - Custom test with N conversations"
        echo ""
        echo "Examples:"
        echo "  $0 quick                    # Quick test"
        echo "  $0 retrieval-only           # Test retrieval only"
        echo "  $0 full                     # Full 500k test"
        echo "  $0 custom 100000            # Custom 100k test"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Test completed!${NC}"
echo -e "${CYAN}Results saved to: tests/scale_test_results.json${NC}"
