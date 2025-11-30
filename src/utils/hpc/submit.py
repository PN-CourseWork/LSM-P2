"""Interactive HPC job submission utilities.

Provides interactive workflows for:
- Selecting and generating job packs
- Previewing and submitting jobs to LSF
"""

import shutil
import subprocess
from pathlib import Path
from typing import List

import questionary

from .jobgen import load_config, generate_pack_lines, write_pack_file


def _get_custom_style():
    """Get custom questionary style."""
    return questionary.Style(
        [
            ("qmark", "fg:#673ab7 bold"),
            ("question", "bold"),
            ("answer", "fg:#f44336 bold"),
            ("pointer", "fg:#673ab7 bold"),
            ("highlighted", "fg:#673ab7 bold"),
            ("selected", "fg:#cc5454"),
            ("separator", "fg:#cc5454"),
            ("instruction", ""),
        ]
    )


def interactive_generate(config_path: Path, job_packs_dir: Path) -> None:
    """Handle job pack generation workflow.

    Parameters
    ----------
    config_path : Path
        Path to the job configuration YAML
    job_packs_dir : Path
        Directory to save generated pack files
    """
    print(f"\n--- Generate Job Pack ---")
    print(f"Loading config: {config_path}")

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    # Extract Group Names based on Schema
    group_names = []
    if "jobs" in config:
        group_names = [j.get("name", "unnamed") for j in config["jobs"]]
    elif "groups" in config:
        group_names = [j.get("name", "unnamed") for j in config["groups"]]
    else:
        group_names = [key for key in config.keys() if key != "defaults"]

    if not group_names:
        print("No job groups found in configuration.")
        return

    # Select Groups
    selected_groups = questionary.checkbox(
        "Select experiment groups to generate packs for "
        "(Space to select, Enter to confirm):",
        choices=group_names,
        style=_get_custom_style(),
    ).ask()

    if not selected_groups:
        print("No groups selected. returning to menu.")
        return

    generated_files = []
    total_jobs_generated = 0

    for group_name in selected_groups:
        job_name_base = config_path.stem
        lines = generate_pack_lines(config, job_name_base, [group_name])

        if not lines:
            print(f"  No jobs generated for group '{group_name}'. Skipping.")
            continue

        print(f"\n  Generated {len(lines)} jobs for group '{group_name}'.")
        total_jobs_generated += len(lines)

        output_file = job_packs_dir / f"{group_name}.pack"
        write_pack_file(output_file, lines)
        generated_files.append(output_file)
        print(f"  Pack file saved to: {output_file}")

        # Preview
        print(f"\n  --- Pack Content Preview for {group_name}.pack ---")
        print("  " + "-" * 40)
        for line in lines[:5]:
            print("  " + line)
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 5} more lines)")
        print("  " + "-" * 40)

    if generated_files:
        print(f"\n--- Generation Summary ---")
        print(
            f"Successfully generated {total_jobs_generated} total jobs "
            f"across {len(generated_files)} pack files:"
        )
        for f in generated_files:
            print(f"  ✓ {f.name}")
    else:
        print("\nNo job packs were generated.")

    print("\n")
    input("Press Enter to continue...")


def interactive_submit(job_packs_dir: Path) -> None:
    """Handle job pack submission workflow.

    Parameters
    ----------
    job_packs_dir : Path
        Directory containing pack files
    """
    print(f"\n--- Submit Job Pack ---")

    if not job_packs_dir.exists():
        print(f"Directory not found: {job_packs_dir}")
        return

    # List pack files, sorted by modification time (newest first)
    pack_files = sorted(
        job_packs_dir.glob("*.pack"), key=lambda f: f.stat().st_mtime, reverse=True
    )

    if not pack_files:
        print("No .pack files found.")
        return

    choices = [f.name for f in pack_files]
    choices.append("Cancel")

    selection = questionary.select(
        "Select a job pack to submit:",
        choices=choices,
        style=_get_custom_style(),
    ).ask()

    if selection == "Cancel" or not selection:
        return

    selected_file = job_packs_dir / selection

    # Preview
    print(f"\nSelected: {selected_file}")
    print("-" * 40)
    with open(selected_file, "r") as f:
        head = [next(f) for _ in range(10)]
    for line in head:
        print(line.strip())
    print("...")
    print("-" * 40)

    # Confirm
    if not shutil.which("bsub"):
        print("Error: 'bsub' command not found. Cannot submit.")
        input("Press Enter to continue...")
        return

    should_submit = questionary.confirm(
        f"Submit {selection} to LSF?",
        default=False,
        style=_get_custom_style(),
    ).ask()

    if should_submit:
        print(f"Submitting {selection}...")
        try:
            subprocess.run(["bsub", "-pack", str(selected_file)], check=True)
            print("Successfully submitted.")
        except subprocess.CalledProcessError as e:
            print(f"Submission failed: {e}")
    else:
        print("Submission cancelled.")

    input("Press Enter to continue...")
