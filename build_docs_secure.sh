#!/bin/bash
#
# Secure Documentation Build Script for Metal-Q
#
# This script builds the documentation in a sandboxed environment with security checks.
# Usage: ./build_docs_secure.sh [options]
#
# Options:
#   --serve       Start a local server after building
#   --validate    Only validate configuration without building
#   --output DIR  Specify output directory (default: site)
#   --audit FILE  Export audit log to specified file
#   --help        Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="site"
SERVE=false
VALIDATE_ONLY=false
AUDIT_FILE=""

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_help() {
    head -20 "$0" | grep '^#' | sed 's/^# *//' | sed 's/^#//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --serve)
            SERVE=true
            shift
            ;;
        --validate)
            VALIDATE_ONLY=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --audit)
            AUDIT_FILE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check Python availability
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

print_info "Metal-Q Secure Documentation Builder"
print_info "====================================="

# Check if we're in the right directory
if [ ! -f "mkdocs.yml" ]; then
    print_error "mkdocs.yml not found. Please run this script from the repository root."
    exit 1
fi

# Install dependencies if needed
print_info "Checking dependencies..."
python3 -c "import yaml" 2>/dev/null || {
    print_warning "Installing PyYAML..."
    pip3 install --user pyyaml
}

# Validation only mode
if [ "$VALIDATE_ONLY" = true ]; then
    print_info "Running configuration validation..."

    if python3 docs/secure_build.py --validate-only; then
        print_info "✓ Configuration validation passed"
        exit 0
    else
        print_error "✗ Configuration validation failed"
        exit 1
    fi
fi

# Pre-build security checks
print_info "Running pre-build security checks..."

# Check for suspicious patterns in documentation
if grep -r "<script" docs/ --include="*.md" 2>/dev/null | grep -v "```"; then
    print_warning "Script tags found in documentation files"
fi

if grep -r "javascript:" docs/ --include="*.md" 2>/dev/null; then
    print_warning "JavaScript protocol links found in documentation"
fi

# Check MkDocs configuration
print_info "Validating MkDocs configuration..."
python3 docs/secure_build.py --validate-only || {
    print_error "Configuration validation failed"
    exit 1
}

# Build documentation in sandbox
print_info "Building documentation in sandboxed environment..."
print_info "Output directory: $OUTPUT_DIR"

BUILD_CMD="python3 docs/secure_build.py --output $OUTPUT_DIR"

if [ -n "$AUDIT_FILE" ]; then
    BUILD_CMD="$BUILD_CMD --export-audit $AUDIT_FILE"
fi

if $BUILD_CMD; then
    print_info "✓ Documentation built successfully"
else
    print_error "✗ Documentation build failed"
    exit 1
fi

# Post-build verification
print_info "Running post-build verification..."

# Check that output was created
if [ ! -d "$OUTPUT_DIR" ]; then
    print_error "Output directory not created"
    exit 1
fi

# Check for sensitive information
if grep -r "PRIVATE KEY" "$OUTPUT_DIR" 2>/dev/null; then
    print_error "Private keys found in built documentation!"
    exit 1
fi

if grep -r "password\s*=\s*['\"]" "$OUTPUT_DIR" 2>/dev/null | grep -v "example"; then
    print_warning "Potential passwords found in documentation"
fi

# Check file permissions
EXEC_FILES=$(find "$OUTPUT_DIR" -type f -perm /111 2>/dev/null | wc -l)
if [ "$EXEC_FILES" -gt 0 ]; then
    print_warning "Found $EXEC_FILES executable files in output"
fi

print_info "✓ Post-build verification complete"

# Export audit log if requested
if [ -n "$AUDIT_FILE" ]; then
    if [ -f "docs_security_audit.log" ]; then
        cp docs_security_audit.log "$AUDIT_FILE"
        print_info "Audit log exported to: $AUDIT_FILE"
    fi
fi

# Generate security report
REPORT_FILE="$OUTPUT_DIR/security-report.txt"
cat > "$REPORT_FILE" << EOF
Documentation Security Build Report
====================================

Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Repository: $(pwd)
Output Directory: $OUTPUT_DIR

Security Checks Performed:
✓ Configuration validation
✓ Path traversal prevention
✓ Sandboxed build environment
✓ Post-build verification
✓ Sensitive data scanning

Build Status: SUCCESS

Recommendations:
- Enable CSP headers when hosting
- Use HTTPS for all documentation URLs
- Regularly update dependencies
- Review audit logs periodically
EOF

print_info "Security report saved to: $REPORT_FILE"

# Serve documentation if requested
if [ "$SERVE" = true ]; then
    print_info "Starting local documentation server..."
    print_info "Documentation will be available at http://localhost:8000"
    print_warning "This is for local testing only. Do not expose to internet."

    cd "$OUTPUT_DIR"
    python3 -m http.server 8000
fi

print_info ""
print_info "============================================"
print_info "✓ Secure documentation build complete!"
print_info "  Output: $OUTPUT_DIR"
print_info "  Files: $(find "$OUTPUT_DIR" -type f | wc -l)"
print_info "  Size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
print_info "============================================"