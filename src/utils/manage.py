"""Project management utilities for running scripts, building docs, and cleaning."""

import subprocess
import shutil
import sys
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any

def get_repo_root() -> Path:
    """Find the project root directory (where pyproject.toml is)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent # Fallback


def load_project_config() -> Dict[str, Any]:
    """Load project_config.yaml if it exists."""
    repo_root = get_repo_root()
    config_path = repo_root / "project_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}



def discover_scripts(pattern: str) -> List[Path]:
    """Find scripts in Experiments/ directory matching pattern."""
    repo_root = get_repo_root()
    experiments_dir = repo_root / "Experiments"

    if not experiments_dir.exists():
        return []

    scripts = [
        p
        for p in experiments_dir.rglob("*.py")
        if p.is_file() and pattern in p.name and p.name != "__init__.py"
    ]

    return sorted(scripts)


def _run_single_script(script: Path, repo_root: Path, timeout: int = 180) -> Tuple[Path, bool, str]:
    """Run a single script and return its result."""
    display_path = script.relative_to(repo_root)

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )

        if result.returncode == 0:
            return (display_path, True, None)
        else:
            error_msg = result.stderr[:200] if result.stderr else ""
            return (display_path, False, f"exit {result.returncode}: {error_msg}")

    except subprocess.TimeoutExpired:
        return (display_path, False, "timeout")
    except Exception as e:
        return (display_path, False, str(e))


def run_plot_scripts():
    """Run plotting scripts in parallel and report results."""
    repo_root = get_repo_root()
    scripts = discover_scripts("plot")

    if not scripts:
        print("  No plot scripts found")
        return

    print(f"\nRunning {len(scripts)} plot scripts in parallel...\n")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor() as executor:
        future_to_script = {
            executor.submit(_run_single_script, script, repo_root): script
            for script in scripts
        }

        for future in as_completed(future_to_script):
            display_path, success, error_msg = future.result()

            if success:
                print(f"  ✓ {display_path}")
                success_count += 1
            else:
                print(f"  ✗ {display_path} ({error_msg})")
                fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")


def run_compute_scripts():
    """Run compute scripts sequentially."""
    repo_root = get_repo_root()
    scripts = discover_scripts("compute")

    if not scripts:
        print("  No compute scripts found")
        return

    print(f"\nRunning {len(scripts)} compute scripts sequentially...\n")

    success_count = 0
    fail_count = 0

    for script in scripts:
        display_path = script.relative_to(repo_root)
        print(f"  → {display_path}...", end=" ", flush=True)

        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for compute scripts
                cwd=str(repo_root),
            )

            if result.returncode == 0:
                print("✓")
                success_count += 1
            else:
                error_msg = (
                    result.stderr[:200]
                    if result.stderr
                    else f"exit {result.returncode}"
                )
                print(f"✗ ({error_msg})")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print("✗ (timeout)")
            fail_count += 1
        except Exception as e:
            print(f"✗ ({e})")
            fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")


def copy_plots():
    """Copy figures/ directory to docs/reports/TexReport/."""
    repo_root = get_repo_root()
    source_dir = repo_root / "figures"
    dest_dir = repo_root / "docs" / "reports" / "TexReport" / "figures"

    print("\nCopying figures/ to docs/reports/TexReport/...")

    if not source_dir.exists():
        print("  No figures/ directory found")
        return

    try:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        print("  ✓ Copied figures/ to docs/reports/TexReport/figures/")
    except Exception as e:
        print(f"  ✗ Failed to copy: {e}")

    print()


def build_docs() -> bool:
    """Build Sphinx documentation."""
    repo_root = get_repo_root()
    docs_dir = repo_root / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    print("\nBuilding Sphinx documentation...")

    if not source_dir.exists():
        print(f"  Error: Documentation source directory not found: {source_dir}")
        return False

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "sphinx-build",
                "-M",
                "html",
                str(source_dir),
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(repo_root),
        )

        if result.returncode == 0:
            print("  ✓ Documentation built successfully")
            print(f"  → Open: {build_dir / 'html' / 'index.html'}\n")
            return True
        else:
            print(f"  ✗ Documentation build failed (exit {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ✗ Documentation build timed out")
        return False
    except FileNotFoundError:
        print("  ✗ sphinx-build not found. Install with: uv sync")
        return False
    except Exception as e:
        print(f"  ✗ Documentation build failed: {e}")
        return False


def clean_all():
    """Clean all generated files and caches."""
    repo_root = get_repo_root()
    print("\nCleaning all generated files and caches...")

    def remove_item(path):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True, None
        except Exception as e:
            return False, str(e)

    cleaned, failed = 0, 0

    # Directories to clean
    dirs = [
        "docs/build",
        "docs/source/example_gallery",
        "docs/source/generated",
        "docs/source/gen_modules",
        "plots",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    ]
    for d in dirs:
        path = repo_root / d
        if path.exists():
            success, _ = remove_item(path)
            cleaned += success
            failed += not success

    # Specific files to clean
    files = ["docs/source/sg_execution_times.rst"]
    for f in files:
        path = repo_root / f
        if path.exists():
            success, _ = remove_item(path)
            cleaned += success
            failed += not success

    # Recursive patterns to clean
    patterns = ["__pycache__", "*.pyc", ".DS_Store"]
    for pattern in patterns:
        for path in repo_root.rglob(pattern):
            success, _ = remove_item(path)
            cleaned += success
            failed += not success

    # Clean data/ directory contents (preserve README.md and .gitkeep)
    data_dir = repo_root / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name not in {"README.md", ".gitkeep"}:
                success, _ = remove_item(item)
                cleaned += success
                failed += not success

    # Clean Experiments/*/output directories
    for output_dir in (repo_root / "Experiments").glob("*/output"):
        success, _ = remove_item(output_dir)
        cleaned += success
        failed += not success

    if cleaned:
        print(f"  ✓ Cleaned {cleaned} items")
    if failed:
        print(f"  ✗ Failed to clean {failed} items")
    if not cleaned and not failed:
        print("  Nothing to clean")
    print()
