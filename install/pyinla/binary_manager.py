"""Binary download and management for pyINLA.

This module handles downloading INLA binaries on-demand instead of bundling them in wheels.
Users can select which binary version to use.
"""

import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional


class BinaryManager:
    """Manages INLA binary downloads and installation."""

    # Binary download URLs - using R-INLA's official download server
    # Same server that R-INLA package uses
    # Environment variable override: PYINLA_BINARY_BASE_URL
    BINARY_URLS = {
        "linux": {
            "23.05.30-1": "https://inla.r-inla-download.org/Linux-builds/Version_23.05.30-1/Ubuntu-22.04/64bit.tgz",
            "latest": "https://inla.r-inla-download.org/Linux-builds/testing/Ubuntu-22.04/64bit.tgz",
        },
        "mac": {
            "23.05.30-1": "https://inla.r-inla-download.org/Mac/testing/Mac-11/64bit.tgz",
            "latest": "https://inla.r-inla-download.org/Mac/testing/Mac-11/64bit.tgz",
        },
        "mac.arm64": {
            "23.05.30-1": "https://inla.r-inla-download.org/Mac/testing/Mac-arm64/64bit.tgz",
            "latest": "https://inla.r-inla-download.org/Mac/testing/Mac-arm64/64bit.tgz",
        },
    }

    def __init__(self):
        """Initialize binary manager."""
        self.package_dir = Path(__file__).parent
        self.bin_dir = self.package_dir / "bin"

    def _detect_bits(self) -> str:
        """Detect if 32 or 64-bit Python."""
        return "64" if sys.maxsize > 2**32 else "32"

    def detect_platform(self) -> str:
        """Detect current platform."""
        system = platform.system().lower()

        if system == "linux":
            return "linux"
        elif system == "darwin":
            # Check if ARM or Intel
            machine = platform.machine().lower()
            if "arm" in machine or "aarch64" in machine:
                return "mac.arm64"
            else:
                return "mac"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

    def get_binary_path(self, platform_name: Optional[str] = None) -> Path:
        """Get path where binary should be installed.

        This must match the paths expected by inla_call.py:
        - Linux: bin/linux/64bit/inla.mkl.run
        - Mac Intel: bin/mac/64bit/inla.run
        - Mac ARM: bin/mac.arm64/inla.run
        """
        if platform_name is None:
            platform_name = self.detect_platform()

        if platform_name == "linux":
            bits = self._detect_bits()
            return self.bin_dir / "linux" / f"{bits}bit" / "inla.mkl.run"
        elif platform_name == "mac":
            bits = self._detect_bits()
            return self.bin_dir / "mac" / f"{bits}bit" / "inla.run"
        elif platform_name == "mac.arm64":
            return self.bin_dir / "mac.arm64" / "inla.run"
        else:
            raise ValueError(f"Unknown platform: {platform_name}")

    def is_installed(self, platform_name: Optional[str] = None) -> bool:
        """Check if binary is already installed."""
        binary_path = self.get_binary_path(platform_name)
        return binary_path.exists() and binary_path.is_file()

    def download_binary(
        self,
        version: str = "latest",
        platform_name: Optional[str] = None,
        force: bool = False
    ) -> Path:
        """Download INLA binary.

        Args:
            version: Binary version to download (default: "latest")
            platform_name: Platform name (auto-detected if None)
            force: Force re-download even if already exists

        Returns:
            Path to downloaded binary
        """
        if platform_name is None:
            platform_name = self.detect_platform()

        binary_path = self.get_binary_path(platform_name)

        # Check if already installed
        if not force and self.is_installed(platform_name):
            print(f"INLA binary already installed at: {binary_path}")
            return binary_path

        # Get download URL
        if platform_name not in self.BINARY_URLS:
            raise ValueError(f"No binary available for platform: {platform_name}")

        platform_versions = self.BINARY_URLS[platform_name]
        if version not in platform_versions:
            available = ", ".join(platform_versions.keys())
            raise ValueError(f"Version '{version}' not available. Available: {available}")

        url = platform_versions[version]

        # Allow custom base URL via environment variable
        custom_base_url = os.environ.get("PYINLA_BINARY_BASE_URL")
        if custom_base_url:
            # Replace the base URL while keeping the path structure
            # Example: https://binaries.pyinla.org/latest/linux/inla.mkl.run
            # becomes: https://your-cdn.com/latest/linux/inla.mkl.run
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path  # /latest/linux/inla.mkl.run
            url = custom_base_url.rstrip('/') + path
            print(f"Using custom binary URL: {url}")

        # Create directory
        binary_path.parent.mkdir(parents=True, exist_ok=True)

        # Download
        print(f"Downloading INLA binary from {url}...")
        print(f"Destination: {binary_path}")

        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192

                with open(binary_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='', flush=True)

                print()  # New line after progress

            # Make executable
            binary_path.chmod(0o755)

            print(f"✓ Binary downloaded and installed successfully")
            return binary_path

        except Exception as e:
            # Clean up partial download
            if binary_path.exists():
                binary_path.unlink()
            raise RuntimeError(f"Failed to download binary: {e}")

    def list_available_versions(self, platform_name: Optional[str] = None) -> list:
        """List available binary versions for a platform."""
        if platform_name is None:
            platform_name = self.detect_platform()

        if platform_name not in self.BINARY_URLS:
            return []

        return list(self.BINARY_URLS[platform_name].keys())

    def remove_binary(self, platform_name: Optional[str] = None) -> None:
        """Remove installed binary."""
        if platform_name is None:
            platform_name = self.detect_platform()

        binary_path = self.get_binary_path(platform_name)

        if binary_path.exists():
            binary_path.unlink()
            print(f"Removed binary: {binary_path}")
        else:
            print(f"No binary found at: {binary_path}")


# Global instance
_manager = BinaryManager()


def download_binary(version: str = "latest", platform: Optional[str] = None, force: bool = False) -> Path:
    """Download INLA binary for current or specified platform.

    Args:
        version: Binary version (default: "latest")
        platform: Platform name (auto-detected if None): "linux", "mac", "mac.arm64"
        force: Force re-download even if exists

    Returns:
        Path to binary

    Example:
        >>> from pyinla import download_binary
        >>> download_binary()  # Downloads latest for current platform
        >>> download_binary(version="23.05.30-1", platform="linux")
    """
    return _manager.download_binary(version=version, platform_name=platform, force=force)


def list_available_binaries(platform: Optional[str] = None) -> list:
    """List available binary versions.

    Args:
        platform: Platform name (auto-detected if None)

    Returns:
        List of available versions
    """
    return _manager.list_available_versions(platform_name=platform)


def is_binary_installed(platform: Optional[str] = None) -> bool:
    """Check if binary is installed for current or specified platform."""
    return _manager.is_installed(platform_name=platform)


def ensure_binary() -> Path:
    """Ensure binary is installed, download if missing.

    This is called automatically on first use of pyinla.

    Returns:
        Path to binary
    """
    if not _manager.is_installed():
        print("INLA binary not found. Downloading...")
        return _manager.download_binary()
    return _manager.get_binary_path()
