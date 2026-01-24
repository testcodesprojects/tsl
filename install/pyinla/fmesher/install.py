"""Helper functions to install R dependencies for fmesher."""

from typing import List, Optional, Tuple


def install_r_packages(
    packages: Optional[List[str]] = None,
    repos: str = "https://cloud.r-project.org",
) -> Tuple[bool, str]:
    """Install required R packages for fmesher.

    Parameters
    ----------
    packages : list of str, optional
        R packages to install. Default: ["fmesher", "sf"]
    repos : str, optional
        CRAN mirror URL.

    Returns
    -------
    tuple
        (success, message)

    Examples
    --------
    >>> from fmesher import install_r_packages
    >>> success, msg = install_r_packages()
    >>> print(msg)
    """
    if packages is None:
        packages = ["fmesher", "sf"]

    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError:
        return False, (
            "rpy2 is not installed. Install with:\n"
            "  pip install rpy2"
        )

    utils = importr("utils")

    messages = []
    all_success = True

    for pkg in packages:
        try:
            # Check if already installed
            try:
                importr(pkg)
                messages.append(f"  {pkg}: already installed")
                continue
            except Exception:
                pass

            # Install package
            messages.append(f"  {pkg}: installing...")
            utils.install_packages(pkg, repos=repos, quiet=True)

            # Verify installation
            try:
                importr(pkg)
                messages.append(f"  {pkg}: installed successfully")
            except Exception as e:
                messages.append(f"  {pkg}: installation failed - {e}")
                all_success = False

        except Exception as e:
            messages.append(f"  {pkg}: error - {e}")
            all_success = False

    status = "All packages ready!" if all_success else "Some packages failed to install"
    return all_success, f"{status}\n" + "\n".join(messages)


def check_r_installation() -> Tuple[bool, str]:
    """Check if R and required packages are properly installed.

    Returns
    -------
    tuple
        (all_ok, detailed_message)
    """
    lines = ["R Installation Check:", "=" * 40]
    all_ok = True

    # Check rpy2
    try:
        import rpy2.robjects as ro
        lines.append("rpy2: OK")
    except ImportError as e:
        lines.append(f"rpy2: MISSING - pip install rpy2")
        all_ok = False
        return all_ok, "\n".join(lines)

    # Check R version
    try:
        r_version = ro.r('R.version.string')[0]
        lines.append(f"R: {r_version}")
    except Exception as e:
        lines.append(f"R: ERROR - {e}")
        all_ok = False

    # Check R packages
    from rpy2.robjects.packages import importr

    r_packages = ["fmesher", "sf"]
    for pkg in r_packages:
        try:
            importr(pkg)
            # Get version
            try:
                ver = ro.r(f'packageVersion("{pkg}")')[0]
                lines.append(f"{pkg}: OK (v{ver})")
            except Exception:
                lines.append(f"{pkg}: OK")
        except Exception:
            lines.append(f"{pkg}: MISSING - install.packages('{pkg}')")
            all_ok = False

    lines.append("=" * 40)
    if all_ok:
        lines.append("All dependencies satisfied!")
    else:
        lines.append("Run: from fmesher import install_r_packages; install_r_packages()")

    return all_ok, "\n".join(lines)
