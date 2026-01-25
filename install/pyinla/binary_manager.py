"""Binary download and management for pyINLA.

This module handles downloading INLA binaries on-demand, similar to R-INLA's
inla.binary.install() function. It fetches available binaries from the
official R-INLA download server.

Linux: Uses https://inla.r-inla-download.org/Linux-builds/FILES listing
Mac: Extracts from R package at https://inla.r-inla-download.org/R/testing/
"""

import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict


class BinaryManager:
    """Manages INLA binary downloads and installation."""

    # Base URLs for R-INLA binary downloads
    LINUX_FILES_URL = "https://inla.r-inla-download.org/Linux-builds/FILES"
    LINUX_BASE_URL = "https://inla.r-inla-download.org/Linux-builds"

    # Mac R packages (binaries are inside these)
    MAC_ARM64_BASE = "https://inla.r-inla-download.org/R/testing/bin/macosx/big-sur-arm64/contrib/4.4"
    MAC_X86_BASE = "https://inla.r-inla-download.org/R/testing/bin/macosx/big-sur-x86_64/contrib/4.4"

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

    def _fetch_linux_files_list(self) -> List[str]:
        """Fetch the FILES listing from R-INLA Linux builds server."""
        try:
            with urllib.request.urlopen(self.LINUX_FILES_URL, timeout=30) as response:
                content = response.read().decode('utf-8')
                return [line.strip() for line in content.split('\n') if line.strip()]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Linux builds list: {e}")

    def list_linux_binaries(self, version: Optional[str] = None, arch: Optional[str] = None) -> List[Dict]:
        """List available Linux binaries.

        Args:
            version: Filter by INLA version (e.g., "26.01.23")
            arch: Filter by architecture ("x86_64" or "aarch64")

        Returns:
            List of dicts with 'os', 'version', 'path', 'url' keys
        """
        files = self._fetch_linux_files_list()

        # Detect current architecture
        if arch is None:
            machine = platform.machine().lower()
            if "aarch64" in machine or "arm64" in machine:
                arch = "aarch64"
            else:
                arch = "x86_64"

        binaries = []
        for line in files:
            # Parse: ./Ubuntu-22.04.5 LTS (Jammy Jellyfish) [x86_64]/Version_26.01.23/64bit.tgz
            if not line.startswith('./') or not line.endswith('/64bit.tgz'):
                continue

            # Filter by architecture
            if arch == "aarch64":
                if "[aarch64]" not in line:
                    continue
            else:
                if "[aarch64]" in line:
                    continue

            # Extract OS and version
            parts = line[2:].split('/')  # Remove ./
            if len(parts) >= 3:
                os_name = parts[0]
                ver_part = parts[1]
                if ver_part.startswith('Version_'):
                    ver = ver_part[8:]  # Remove 'Version_'

                    # Filter by version if specified
                    if version and version not in ver:
                        continue

                    # Build URL
                    url_path = urllib.parse.quote(line[2:], safe='/')
                    url = f"{self.LINUX_BASE_URL}/{url_path}"

                    binaries.append({
                        'os': os_name,
                        'version': ver,
                        'path': line,
                        'url': url
                    })

        return binaries

    def _download_file(self, url: str, dest: Path, desc: str = "Downloading") -> None:
        """Download a file with progress display."""
        print(f"{desc}: {url}")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192

                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\rProgress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end='', flush=True)

                print()  # New line after progress

        except Exception as e:
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Download failed: {e}")

    def download_linux_binary(
        self,
        os_name: Optional[str] = None,
        version: Optional[str] = None,
        force: bool = False,
        interactive: bool = True
    ) -> Path:
        """Download Linux INLA binary.

        Args:
            os_name: OS name to filter (e.g., "Ubuntu-22.04"). If None, show choices.
            version: Version to download. If None, use latest.
            force: Force re-download even if exists
            interactive: If True, prompt user to choose when multiple options

        Returns:
            Path to installed binary
        """
        binary_path = self.get_binary_path("linux")

        if not force and self.is_installed("linux"):
            print(f"INLA binary already installed at: {binary_path}")
            return binary_path

        # Get available binaries
        binaries = self.list_linux_binaries(version=version)

        if not binaries:
            raise RuntimeError("No Linux binaries found. Check your internet connection.")

        # Filter by OS if specified
        if os_name:
            filtered = [b for b in binaries if os_name.lower() in b['os'].lower()]
            if not filtered:
                # Show available options to help user
                available_os = sorted(set(b['os'] for b in binaries))[:10]
                os_examples = '\n  '.join(available_os[:5])
                raise RuntimeError(
                    f"No binaries found matching '{os_name}'.\n\n"
                    f"Available OS options (showing first 5):\n  {os_examples}\n\n"
                    f"Use list_available_os() to see all options, or call download_binary() "
                    f"without os_name for interactive selection."
                )
            binaries = filtered

        # Sort by version (newest first) and OS
        binaries.sort(key=lambda x: (x['version'], x['os']), reverse=True)

        # If multiple options and interactive, let user choose
        if len(binaries) > 1 and interactive and os_name is None:
            print("\nAvailable Linux binaries:")
            # Show unique OS options for latest version
            latest_version = binaries[0]['version']
            latest_binaries = [b for b in binaries if b['version'] == latest_version]

            for i, b in enumerate(latest_binaries, 1):
                print(f"  {i}. {b['os']} (Version {b['version']})")

            print(f"\nRecommended: Ubuntu-22.04 or Ubuntu-24.04 for most systems")

            try:
                choice = input(f"Choose [1-{len(latest_binaries)}] (or press Enter for first): ").strip()
                if choice == "":
                    idx = 0
                else:
                    idx = int(choice) - 1
                    if idx < 0 or idx >= len(latest_binaries):
                        raise ValueError()
                selected = latest_binaries[idx]
            except (ValueError, EOFError):
                print("Using first option...")
                selected = latest_binaries[0]
        else:
            # Use first match (latest version)
            selected = binaries[0]

        print(f"\nSelected: {selected['os']} Version {selected['version']}")

        # Download to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tgz_path = Path(tmpdir) / "64bit.tgz"
            self._download_file(selected['url'], tgz_path, "Downloading binary")

            # Extract
            print("Extracting...")
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(tmpdir)

            # Find and install binary
            extracted_dir = Path(tmpdir) / "64bit"
            if not extracted_dir.exists():
                # Sometimes it extracts directly
                extracted_dir = Path(tmpdir)

            # Look for inla binary
            inla_binary = None
            for name in ["inla.mkl.run", "inla.run", "inla"]:
                candidate = extracted_dir / name
                if candidate.exists():
                    inla_binary = candidate
                    break

            if not inla_binary:
                raise RuntimeError(f"Could not find INLA binary in extracted files")

            # Install ALL files from extracted directory (not just the wrapper)
            # The .run files are shell scripts that call the actual binaries
            binary_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy all files from extracted directory
            for item in extracted_dir.iterdir():
                if item.is_file():
                    dst = binary_path.parent / item.name
                    shutil.copy2(item, dst)
                    dst.chmod(0o755)
                elif item.is_dir():
                    # Copy subdirectories too (e.g., external libs)
                    dst_dir = binary_path.parent / item.name
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(item, dst_dir)

            print(f"Installed {len(list(extracted_dir.iterdir()))} files to: {binary_path.parent}")

        print(f"Installed INLA binary to: {binary_path}")
        return binary_path

    def download_mac_binary(
        self,
        version: Optional[str] = None,
        force: bool = False
    ) -> Path:
        """Download Mac INLA binary from R package.

        Args:
            version: Version to download (e.g., "25.04.16"). If None, fetches latest.
            force: Force re-download

        Returns:
            Path to installed binary
        """
        platform_name = self.detect_platform()
        if platform_name not in ("mac", "mac.arm64"):
            raise RuntimeError("This function is for Mac only")

        binary_path = self.get_binary_path(platform_name)

        if not force and self.is_installed(platform_name):
            print(f"INLA binary already installed at: {binary_path}")
            return binary_path

        # Select base URL based on architecture
        if platform_name == "mac.arm64":
            base_url = self.MAC_ARM64_BASE
            bin_subdir = "mac.arm64"
        else:
            base_url = self.MAC_X86_BASE
            bin_subdir = "mac/64bit"

        # Find available versions
        print(f"Fetching available Mac packages...")
        try:
            with urllib.request.urlopen(f"{base_url}/", timeout=30) as response:
                content = response.read().decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Mac packages list: {e}")

        # Parse package names
        import re
        packages = re.findall(r'INLA_([0-9.]+)\.tgz', content)
        packages = sorted(set(packages), reverse=True)

        if not packages:
            raise RuntimeError("No Mac packages found")

        # Select version
        if version:
            if version not in packages:
                raise RuntimeError(f"Version {version} not found. Available: {packages[:5]}")
            selected_version = version
        else:
            selected_version = packages[0]  # Latest

        pkg_url = f"{base_url}/INLA_{selected_version}.tgz"
        print(f"Selected: INLA {selected_version}")

        # Download and extract
        with tempfile.TemporaryDirectory() as tmpdir:
            tgz_path = Path(tmpdir) / "INLA.tgz"
            self._download_file(pkg_url, tgz_path, "Downloading Mac package")

            print("Extracting...")
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(tmpdir)

            # Find binary in extracted R package
            extracted = Path(tmpdir) / "INLA" / "bin" / bin_subdir

            inla_binary = None
            for name in ["inla.run", "inla"]:
                candidate = extracted / name
                if candidate.exists():
                    inla_binary = candidate
                    break

            if not inla_binary:
                # Try alternative paths
                for root, dirs, files in os.walk(Path(tmpdir) / "INLA" / "bin"):
                    for f in files:
                        if f in ("inla.run", "inla"):
                            inla_binary = Path(root) / f
                            break
                    if inla_binary:
                        break

            if not inla_binary:
                raise RuntimeError("Could not find INLA binary in Mac package")

            # Install ALL files from the binary directory
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            src_dir = inla_binary.parent

            for item in src_dir.iterdir():
                if item.is_file():
                    dst = binary_path.parent / item.name
                    shutil.copy2(item, dst)
                    dst.chmod(0o755)
                elif item.is_dir():
                    dst_dir = binary_path.parent / item.name
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(item, dst_dir)

        print(f"Installed INLA binary to: {binary_path.parent}")
        return binary_path

    def download_binary(
        self,
        os_name: Optional[str] = None,
        version: Optional[str] = None,
        platform_name: Optional[str] = None,
        force: bool = False,
        interactive: bool = True
    ) -> Path:
        """Download INLA binary for current or specified platform.

        Args:
            os_name: For Linux, specify OS (e.g., "Ubuntu-22.04")
            version: Binary version to download
            platform_name: Platform ("linux", "mac", "mac.arm64"). Auto-detected if None.
            force: Force re-download
            interactive: If True, prompt user to choose for Linux

        Returns:
            Path to installed binary
        """
        if platform_name is None:
            platform_name = self.detect_platform()

        if platform_name == "linux":
            return self.download_linux_binary(
                os_name=os_name,
                version=version,
                force=force,
                interactive=interactive
            )
        elif platform_name in ("mac", "mac.arm64"):
            return self.download_mac_binary(version=version, force=force)
        else:
            raise ValueError(f"Unknown platform: {platform_name}")

    def list_available_versions(self, platform_name: Optional[str] = None) -> List[str]:
        """List available binary versions for a platform."""
        if platform_name is None:
            platform_name = self.detect_platform()

        if platform_name == "linux":
            binaries = self.list_linux_binaries()
            versions = sorted(set(b['version'] for b in binaries), reverse=True)
            return versions[:10]  # Return latest 10
        else:
            # For Mac, would need to parse the package listing
            return ["latest"]

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


def download_binary(
    os_name: Optional[str] = None,
    version: Optional[str] = None,
    platform: Optional[str] = None,
    force: bool = False,
    interactive: bool = True
) -> Path:
    """Download INLA binary for current or specified platform.

    This works like R-INLA's inla.binary.install() function.

    Args:
        os_name: For Linux, specify OS (e.g., "Ubuntu-22.04", "Ubuntu-24.04")
        version: Binary version (e.g., "26.01.23")
        platform: Platform name: "linux", "mac", "mac.arm64" (auto-detected if None)
        force: Force re-download even if exists
        interactive: If True, show menu to choose Linux OS variant

    Returns:
        Path to binary

    Example:
        >>> from pyinla import download_binary
        >>> download_binary()  # Interactive selection for current platform
        >>> download_binary(os_name="Ubuntu-22.04")  # Specific Linux OS
        >>> download_binary(platform="mac.arm64")  # Mac ARM
    """
    return _manager.download_binary(
        os_name=os_name,
        version=version,
        platform_name=platform,
        force=force,
        interactive=interactive
    )


def list_available_binaries(platform: Optional[str] = None) -> List[str]:
    """List available binary versions.

    Args:
        platform: Platform name (auto-detected if None)

    Returns:
        List of available versions
    """
    return _manager.list_available_versions(platform_name=platform)


def list_available_os(print_list: bool = True) -> List[str]:
    """List available OS options for Linux binary download.

    This helps you find the correct os_name string for download_binary().

    Args:
        print_list: If True, print a formatted list (default). If False, just return the list.

    Returns:
        List of available OS names (e.g., ["Ubuntu-22.04.5 LTS (Jammy Jellyfish) [x86_64]", ...])

    Example:
        >>> from pyinla import list_available_os
        >>> list_available_os()  # Prints formatted list
        Available Linux OS options:
          1. Ubuntu-22.04.5 LTS (Jammy Jellyfish) [x86_64]
          2. Ubuntu-24.04.1 LTS (Noble Numbat) [x86_64]
          ...

        Use with download_binary:
          download_binary(os_name="Ubuntu-22.04")  # partial match works
    """
    binaries = _manager.list_linux_binaries()

    if not binaries:
        if print_list:
            print("Could not fetch available binaries. Check internet connection.")
        return []

    # Get unique OS names from latest version
    latest_version = max(set(b['version'] for b in binaries))
    os_names = sorted(set(b['os'] for b in binaries if b['version'] == latest_version))

    if print_list:
        print(f"\nAvailable Linux OS options (Version {latest_version}):\n")
        for i, os_name in enumerate(os_names, 1):
            print(f"  {i:2}. {os_name}")
        print("\nUsage with download_binary() - partial match works:")
        print('  download_binary(os_name="Ubuntu-22.04")  # matches Ubuntu-22.04.x LTS ...')
        print('  download_binary(os_name="Ubuntu-24")     # matches Ubuntu-24.x LTS ...')
        print('  download_binary(os_name="Fedora")        # matches Fedora-x ...')

    return os_names


def is_binary_installed(platform: Optional[str] = None) -> bool:
    """Check if binary is installed for current or specified platform."""
    return _manager.is_installed(platform_name=platform)


def ensure_binary(os_name: Optional[str] = None) -> Path:
    """Ensure binary is installed, download if missing.

    This is called automatically on first use of pyinla.

    Args:
        os_name: For Linux, specify OS variant (e.g., "Ubuntu-22.04")

    Returns:
        Path to binary
    """
    if not _manager.is_installed():
        print("INLA binary not found. Downloading...")
        return _manager.download_binary(os_name=os_name, interactive=True)
    return _manager.get_binary_path()


def _cli_install():
    """CLI entry point for installing INLA binary.

    Run after pip install:
        pyinla-install
        pyinla-install --os Ubuntu-22.04
        pyinla-install --force
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and install INLA binary for pyINLA",
        prog="pyinla-install"
    )
    parser.add_argument(
        "--os",
        help="Linux OS variant (e.g., Ubuntu-22.04, Ubuntu-24.04, Fedora-40)"
    )
    parser.add_argument(
        "--version",
        help="INLA version (e.g., 26.01.23)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if binary exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available OS options for Linux"
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List available binary versions"
    )

    args = parser.parse_args()

    if args.list:
        platform_name = _manager.detect_platform()
        print(f"Platform: {platform_name}")

        if platform_name == "linux":
            list_available_os(print_list=True)
        else:
            print(f"\nMac detected ({platform_name}) - binary auto-selected based on architecture.")
            print("Just run: pyinla-install")
        return

    if args.list_versions:
        platform_name = _manager.detect_platform()
        print(f"Platform: {platform_name}")

        if platform_name == "linux":
            print("\nFetching available Linux binaries...")
            binaries = _manager.list_linux_binaries()

            # Group by version
            versions = {}
            for b in binaries:
                if b['version'] not in versions:
                    versions[b['version']] = []
                versions[b['version']].append(b['os'])

            print(f"\nFound {len(versions)} versions (showing latest 5):")
            for ver in sorted(versions.keys(), reverse=True)[:5]:
                print(f"\n  Version {ver}:")
                for os_name in sorted(versions[ver])[:10]:
                    print(f"    - {os_name}")
        else:
            versions = _manager.list_available_versions()
            print(f"Available versions: {versions}")
        return

    try:
        path = download_binary(
            os_name=args.os,
            version=args.version,
            force=args.force,
            interactive=True
        )
        print(f"\nSuccess! Binary installed at: {path}")
        print("\nYou can now use pyINLA:")
        print("  import pyinla")
        print("  pyinla.activate('YOUR-LICENSE-KEY')")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
