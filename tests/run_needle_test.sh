#!/bin/bash
# Run needle-in-haystack test with environment variables

set -a  # Export all variables
source .env
set +a

# Activate virtual environment
source .venv/bin/activate

# Run the test
python tests/test_needle_haystack.py "$@"
