"""Interactive HPC job submission utilities.

Provides interactive workflows for:
- Selecting and generating job packs
- Previewing and submitting jobs to LSF

Works both standalone (questionary) and within TUI (TuiRunner).
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .jobgen import load_config, generate_pack_lines, write_pack_file

if TYPE_CHECKING:
    from utils.tui.runner import TuiRunner


def generate_pack(
    config_path: Path,
    job_packs_dir: Path,
    selected_groups: List[str],
) -> tuple[List[Path], List[str]]:
    """Generate job pack files for selected groups.

    Parameters
    ----------
    config_path : Path
        Path to the job configuration YAML
    job_packs_dir : Path
        Directory to save generated pack files
    selected_groups : List[str]
        Groups to generate packs for

    Returns
    -------
    tuple[List[Path], List[str]]
        Generated files and log messages
    """
    log = []
    log.append(f"Loading config: {config_path}")

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return [], [f"Error: Config file not found at {config_path}"]

    generated_files = []
    total_jobs = 0

    for group_name in selected_groups:
        job_name_base = config_path.stem
        lines = generate_pack_lines(config, job_name_base, [group_name])

        if not lines:
            log.append(f"No jobs generated for group '{group_name}'. Skipping.")
            continue

        log.append(f"Generated {len(lines)} jobs for group '{group_name}'")
        total_jobs += len(lines)

        output_file = job_packs_dir / f"{group_name}.pack"
        write_pack_file(output_file, lines)
        generated_files.append(output_file)
        log.append(f"Saved: {output_file}")

        # Preview first few lines
        log.append("")
        log.append(f"--- Preview: {group_name}.pack ---")
        for line in lines[:3]:
            # Truncate long lines
            if len(line) > 80:
                log.append(f"  {line[:77]}...")
            else:
                log.append(f"  {line}")
        if len(lines) > 3:
            log.append(f"  ... ({len(lines) - 3} more jobs)")
        log.append("")

    if generated_files:
        log.append("=" * 40)
        log.append(f"Generated {total_jobs} jobs in {len(generated_files)} pack files")
        for f in generated_files:
            log.append(f"  ✓ {f.name}")
    else:
        log.append("No job packs were generated.")

    return generated_files, log


def submit_pack(pack_file: Path) -> tuple[bool, List[str]]:
    """Submit a pack file to LSF.

    Parameters
    ----------
    pack_file : Path
        Path to the pack file

    Returns
    -------
    tuple[bool, List[str]]
        Success status and log messages
    """
    log = []

    if not shutil.which("bsub"):
        return False, ["Error: 'bsub' command not found. Cannot submit."]

    log.append(f"Submitting {pack_file.name}...")

    try:
        result = subprocess.run(
            ["bsub", "-pack", str(pack_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.append("Successfully submitted!")
            if result.stdout:
                log.extend(result.stdout.splitlines())
            return True, log
        else:
            log.append(f"Submission failed (exit code {result.returncode})")
            if result.stderr:
                log.extend(result.stderr.splitlines())
            return False, log
    except Exception as e:
        return False, [f"Submission failed: {e}"]


def get_available_groups(config_path: Path) -> List[str]:
    """Get list of available job groups from config.

    Parameters
    ----------
    config_path : Path
        Path to config file

    Returns
    -------
    List[str]
        Group names
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return []

    if "jobs" in config:
        return [j.get("name", "unnamed") for j in config["jobs"]]
    elif "groups" in config:
        return [j.get("name", "unnamed") for j in config["groups"]]
    else:
        return [key for key in config.keys() if not key.startswith("_")]


def get_available_packs(config_path: Path) -> dict:
    """Get pack definitions from config.

    Packs are top-level keys whose value is a list of group names.
    Groups are top-level keys whose value is a dict with lsf_options/command.

    Parameters
    ----------
    config_path : Path
        Path to config file

    Returns
    -------
    dict
        Pack name -> list of groups
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return {}

    packs = {}
    for key, value in config.items():
        # Packs are lists of strings (group names)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            packs[key] = value
    return packs


def get_pack_files(job_packs_dir: Path) -> List[Path]:
    """Get list of pack files sorted by modification time.

    Parameters
    ----------
    job_packs_dir : Path
        Directory containing pack files

    Returns
    -------
    List[Path]
        Pack files, newest first
    """
    if not job_packs_dir.exists():
        return []

    return sorted(
        job_packs_dir.glob("*.pack"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


# --- TUI-integrated versions ---


def tui_generate_pack(
    runner: "TuiRunner",
    config_path: Path,
    job_packs_dir: Path,
) -> None:
    """Generate job pack with TUI interface.

    Parameters
    ----------
    runner : TuiRunner
        TUI runner instance
    config_path : Path
        Path to config file
    job_packs_dir : Path
        Output directory for pack files
    """
    from utils.tui.runner import SelectOption

    # Get available groups
    groups = get_available_groups(config_path)

    if not groups:
        runner.show_output(
            "Generate Job Pack",
            [f"No job groups found in {config_path}"],
        )
        return

    # Select groups
    options = [SelectOption(label=g, value=g) for g in groups]
    selected = runner.select(
        "Select Groups to Generate",
        options,
        allow_multiple=True,
    )

    if not selected:
        return

    # Generate packs
    files, log = generate_pack(config_path, job_packs_dir, selected)

    # Show results
    runner.show_output("Generation Results", log)


def get_group_config_preview(config_path: Path, group_name: str) -> list[str]:
    """Get formatted preview of a group's YAML configuration.

    Parameters
    ----------
    config_path : Path
        Path to packs.yaml
    group_name : str
        Name of the group to preview

    Returns
    -------
    list[str]
        Formatted lines for display
    """
    lines = []

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return [f"Config not found: {config_path}"]

    # Find the group (top-level key)
    if group_name not in config:
        return [f"Group '{group_name}' not found in config"]

    group_config = config[group_name]

    # Format the config nicely
    lines.append(f"Group: {group_name}")
    lines.append("=" * 50)
    lines.append("")

    # LSF options
    if "lsf_options" in group_config:
        lines.append("LSF Options:")
        for opt in group_config["lsf_options"]:
            lines.append(f"  {opt}")
        lines.append("")

    # Command template
    if "command" in group_config:
        lines.append("Command Template:")
        cmd = group_config["command"]
        # Wrap long commands
        if len(cmd) > 60:
            words = cmd.split()
            current_line = " "
            for word in words:
                if len(current_line) + len(word) > 60:
                    lines.append(current_line)
                    current_line = "   " + word
                else:
                    current_line += " " + word
            if current_line.strip():
                lines.append(current_line)
        else:
            lines.append(f"  {cmd}")
        lines.append("")

    # Static args
    static = group_config.get("static_args", {})
    if static:
        lines.append("Static Args:")
        for key, value in static.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Sweep parameters
    if "sweep" in group_config:
        lines.append("Sweep Parameters:")
        total_combos = 1
        for key, values in group_config["sweep"].items():
            if isinstance(values, list):
                lines.append(f"  {key}: {values}")
                total_combos *= len(values)
            else:
                lines.append(f"  {key}: {values}")
        lines.append("")
        lines.append(f"Total combinations: {total_combos}")

    return lines


def tui_submit_pack(
    runner: "TuiRunner",
    job_packs_dir: Path,
) -> None:
    """Submit job pack with TUI interface.

    Parameters
    ----------
    runner : TuiRunner
        TUI runner instance
    job_packs_dir : Path
        Directory containing pack files
    """
    from utils.tui.runner import SelectOption

    # Get pack files
    pack_files = get_pack_files(job_packs_dir)

    if not pack_files:
        runner.show_output(
            "Submit Job Pack",
            [f"No .pack files found in {job_packs_dir}"],
        )
        return

    # Select pack file
    options = [SelectOption(label=f.name, value=f) for f in pack_files]
    selected = runner.select("Select Pack to Submit", options)

    if not selected:
        return

    # Get group name from pack file name (e.g., "SN_strong.pack" -> "SN_strong")
    group_name = selected.stem

    # Build preview with YAML config + pack file content
    preview = []

    # First show YAML group config
    config_path = job_packs_dir / "packs.yaml"
    if config_path.exists():
        preview.extend(get_group_config_preview(config_path, group_name))
        preview.append("")
        preview.append("─" * 50)
        preview.append("")

    # Then show pack file preview
    with open(selected, "r") as f:
        pack_lines = f.readlines()

    job_count = len([l for l in pack_lines if l.strip() and not l.startswith('#')])

    preview.append(f"Pack File: {selected.name}")
    preview.append(f"Total Jobs: {job_count}")
    preview.append("")
    preview.append("Generated Commands (first 5):")
    preview.append("-" * 30)

    shown = 0
    for line in pack_lines:
        line = line.rstrip()
        if line and not line.startswith('#'):
            # Truncate long lines for display
            if len(line) > 100:
                preview.append(f"  {line[:97]}...")
            else:
                preview.append(f"  {line}")
            shown += 1
            if shown >= 5:
                break

    if job_count > 5:
        preview.append(f"  ... ({job_count - 5} more jobs)")

    runner.show_output("Pack Preview", preview, wait_for_key=True)

    # Confirm submission
    if not runner.confirm(f"Submit {selected.name} ({job_count} jobs) to LSF?"):
        return

    # Submit
    success, log = submit_pack(selected)
    runner.show_output(
        "Submission " + ("Successful" if success else "Failed"),
        log,
    )


# --- Questionary-based interactive versions ---


def _get_questionary_style():
    """Get questionary style without TUI dependency."""
    from questionary import Style
    return Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:green'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
    ])


def interactive_generate(config_path: str | Path, job_packs_dir: Path | None = None) -> None:
    """Generate all job packs from config.

    Parameters
    ----------
    config_path : str or Path
        Path to packs.yaml configuration file
    job_packs_dir : Path, optional
        Output directory for pack files. Defaults to same directory as config.
    """
    from pathlib import Path

    config_path = Path(config_path)
    if job_packs_dir is None:
        job_packs_dir = config_path.parent

    print(f"\n--- Generate Job Packs ---")
    print(f"Config: {config_path}")

    config = load_config(config_path)

    total_jobs = 0
    generated_files = []

    # Each top-level key is a pack, containing groups
    for pack_name, pack_config in config.items():
        if not isinstance(pack_config, dict):
            continue

        lines = generate_pack_lines(pack_config, pack_name)
        output_file = job_packs_dir / f"{pack_name}.pack"
        write_pack_file(output_file, lines)

        job_count = len([l for l in lines if l.strip() and not l.startswith('#')])
        total_jobs += job_count
        group_names = [k for k in pack_config.keys() if isinstance(pack_config[k], dict)]
        generated_files.append((output_file.name, job_count, group_names))
        print(f"  ✓ {output_file.name}: {job_count} jobs")

    print(f"\nGenerated {len(generated_files)} pack files with {total_jobs} total jobs")
    for name, count, groups in generated_files:
        print(f"  {name}: {count} jobs ({', '.join(groups)})")


def interactive_submit(job_packs_dir: Path) -> None:
    """Handle job pack submission workflow."""
    import questionary

    print(f"\n--- Submit Job Pack ---")

    pack_files = get_pack_files(job_packs_dir)
    if not pack_files:
        print(f"No .pack files found in {job_packs_dir}")
        return

    choices = [f.name for f in pack_files] + ["Cancel"]
    selection = questionary.select(
        "Select pack to submit:",
        choices=choices,
        style=_get_questionary_style(),
    ).ask()

    if selection == "Cancel" or not selection:
        return

    selected_file = job_packs_dir / selection

    # Preview
    print(f"\nSelected: {selected_file}")
    with open(selected_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:
                print("...")
                break
            print(line.rstrip())

    if not questionary.confirm("Submit to LSF?", style=_get_questionary_style()).ask():
        return

    success, log = submit_pack(selected_file)
    for line in log:
        print(line)

    input("\nPress Enter to continue...")
