#!/bin/bash
# ---------------------------------------------------------------------------
#  HeartMuLa Studio - Build Asset Validation Script
#  Validates that all required files and dependencies are in place
#  Can run on Linux to check before macOS build
# ---------------------------------------------------------------------------

echo "=========================================="
echo "HeartMuLa Studio - Build Validation"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found directory: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing directory: $1"
        return 1
    fi
}

# Track errors
ERRORS=0

echo "1. Checking required files..."
echo "------------------------------------"

# Build scripts
check_file "build/macos/generate_icon.sh" || ((ERRORS++))
check_file "build/macos/codesign.sh" || ((ERRORS++))
check_file ".github/DMG_README.txt" || ((ERRORS++))

# Icon source
check_file "frontend/public/heartmula-icon.svg" || ((ERRORS++))

# PyInstaller spec
check_file "HeartMuLa.spec" || ((ERRORS++))

# Requirements files
check_file "requirements_macos.txt" || ((ERRORS++))
check_file "backend/requirements.txt" || ((ERRORS++))

echo ""
echo "2. Checking required directories..."
echo "------------------------------------"

# Backend directories (referenced in HeartMuLa.spec)
check_dir "backend" || ((ERRORS++))
check_dir "backend/models" || ((ERRORS++))
check_dir "backend/generated_audio" || ((ERRORS++))
check_dir "backend/ref_audio" || ((ERRORS++))
check_dir "backend/app" || ((ERRORS++))

# Frontend
check_dir "frontend" || ((ERRORS++))
check_dir "frontend/public" || ((ERRORS++))

# Build directories
check_dir "build/macos" || ((ERRORS++))
check_dir ".github" || ((ERRORS++))

echo ""
echo "3. Validating requirements_macos.txt..."
echo "------------------------------------"

# Check for critical dependencies
if grep -q "transformers" requirements_macos.txt; then
    if grep -q "transformers.*<4.57.0" requirements_macos.txt; then
        echo -e "${GREEN}✓${NC} transformers version constrained (avoiding yanked 4.57.0)"
    else
        echo -e "${YELLOW}⚠${NC} transformers found but version not constrained"
        echo "   Recommendation: Add constraint <4.57.0 to avoid yanked version"
    fi
else
    echo -e "${YELLOW}⚠${NC} transformers not explicitly listed (may come from heartlib)"
fi

if grep -q "triton" requirements_macos.txt; then
    echo -e "${GREEN}✓${NC} triton listed (optional for torch.compile)"
else
    echo -e "${YELLOW}⚠${NC} triton not listed (optional but recommended for torch.compile)"
fi

if grep -q "heartlib" requirements_macos.txt; then
    echo -e "${GREEN}✓${NC} heartlib dependency present"
else
    echo -e "${RED}✗${NC} heartlib dependency missing"
    ((ERRORS++))
fi

if grep -q "pyinstaller" requirements_macos.txt; then
    echo -e "${GREEN}✓${NC} pyinstaller dependency present"
else
    echo -e "${RED}✗${NC} pyinstaller dependency missing"
    ((ERRORS++))
fi

echo ""
echo "4. Validating HeartMuLa.spec..."
echo "------------------------------------"

# Check that spec file references correct directories
if grep -q "backend/models" HeartMuLa.spec; then
    echo -e "${GREEN}✓${NC} HeartMuLa.spec includes backend/models"
else
    echo -e "${RED}✗${NC} HeartMuLa.spec missing backend/models reference"
    ((ERRORS++))
fi

if grep -q "frontend/dist" HeartMuLa.spec; then
    echo -e "${GREEN}✓${NC} HeartMuLa.spec includes frontend/dist"
else
    echo -e "${RED}✗${NC} HeartMuLa.spec missing frontend/dist reference"
    ((ERRORS++))
fi

if grep -q "build/macos/HeartMuLa.icns" HeartMuLa.spec; then
    echo -e "${GREEN}✓${NC} HeartMuLa.spec references icon file"
else
    echo -e "${RED}✗${NC} HeartMuLa.spec missing icon reference"
    ((ERRORS++))
fi

echo ""
echo "5. Checking script permissions..."
echo "------------------------------------"

if [ -x "build/macos/generate_icon.sh" ]; then
    echo -e "${GREEN}✓${NC} generate_icon.sh is executable"
else
    echo -e "${YELLOW}⚠${NC} generate_icon.sh not executable (will be set during build)"
fi

if [ -x "build/macos/codesign.sh" ]; then
    echo -e "${GREEN}✓${NC} codesign.sh is executable"
else
    echo -e "${YELLOW}⚠${NC} codesign.sh not executable (will be set during build)"
fi

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All validation checks passed!${NC}"
    echo "=========================================="
    echo ""
    echo "Ready for macOS build workflow:"
    echo "  - All required files are present"
    echo "  - Directory structure is correct"
    echo "  - Dependencies are properly configured"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    echo "=========================================="
    echo ""
    echo "Please fix the errors above before running the build."
    exit 1
fi
