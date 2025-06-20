#!/bin/bash
# Integration test runner for T2L Core

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TEST_THREADS=1
NOCAPTURE=false
SPECIFIC_TEST=""
LOG_LEVEL="info"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threads NUM     Number of test threads (default: 1)"
    echo "  -n, --nocapture      Don't capture test output"
    echo "  -s, --specific TEST   Run specific test module"
    echo "  -l, --log-level LEVEL Set log level (default: info)"
    echo "  -b, --benchmarks      Run only benchmark tests"
    echo "  -q, --quick           Run quick tests only (exclude benchmarks)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Test modules:"
    echo "  workflow_tests    - Full workflow integration tests"
    echo "  cli_tests        - CLI command tests"
    echo "  export_tests     - Export format compatibility tests"
    echo "  architecture_tests - Architecture compatibility tests"
    echo "  error_tests      - Error handling tests"
    echo "  performance_tests - Performance benchmarks"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            TEST_THREADS="$2"
            shift 2
            ;;
        -n|--nocapture)
            NOCAPTURE=true
            shift
            ;;
        -s|--specific)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -b|--benchmarks)
            SPECIFIC_TEST="performance_tests"
            NOCAPTURE=true
            shift
            ;;
        -q|--quick)
            SPECIFIC_TEST="workflow_tests cli_tests export_tests"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Build the cargo test command
CMD="cargo test --test integration"

# Add specific test if provided
if [ -n "$SPECIFIC_TEST" ]; then
    CMD="$CMD $SPECIFIC_TEST"
fi

# Add test threads
CMD="$CMD -- --test-threads=$TEST_THREADS"

# Add nocapture if requested
if [ "$NOCAPTURE" = true ]; then
    CMD="$CMD --nocapture"
fi

# Set environment variables
export RUST_LOG="t2l_core=$LOG_LEVEL"
export RUST_BACKTRACE=1

# Print test configuration
echo -e "${GREEN}Running T2L Integration Tests${NC}"
echo "Configuration:"
echo "  Test threads: $TEST_THREADS"
echo "  Log level: $LOG_LEVEL"
echo "  Capture output: $([ "$NOCAPTURE" = true ] && echo "no" || echo "yes")"
[ -n "$SPECIFIC_TEST" ] && echo "  Specific tests: $SPECIFIC_TEST"
echo ""

# Run the tests
echo -e "${YELLOW}Executing: $CMD${NC}"
echo ""

if $CMD; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi