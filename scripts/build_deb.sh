#!/bin/bash
set -e
VERSION=${1:-0.1.0}
PKGDIR=$(mktemp -d)
install -Dm755 crmd.py "$PKGDIR/usr/bin/crmd"
cat > "$PKGDIR/DEBIAN/control" <<CTL
Package: crmd
Version: $VERSION
Section: utils
Priority: optional
Architecture: all
Maintainer: Your Name
Description: Terminal CRM with OpenAI integration
CTL
dpkg-deb --build "$PKGDIR" "crmd_${VERSION}_all.deb"
rm -rf "$PKGDIR"
