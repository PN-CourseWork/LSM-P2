"""Live HPC job monitor TUI using blessed."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from blessed import Terminal


@dataclass
class Job:
    """HPC job info."""
    id: str
    name: str
    queue: str
    status: str
    cores: str
    start_time: str
    elapsed: str
    output_file: str = ""


def get_jobs() -> list[Job]:
    """Fetch current jobs from bstat (preferred) or bjobs."""
    try:
        result = subprocess.run(["bstat"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 9:
                    jobs.append(Job(
                        id=parts[0],
                        queue=parts[2],
                        name=parts[3],
                        cores=parts[4],
                        status=parts[5],
                        start_time=f"{parts[6]} {parts[7]} {parts[8]}",
                        elapsed=parts[-1],
                    ))
            return jobs
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["bjobs", "-w"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 7:
                    jobs.append(Job(
                        id=parts[0],
                        status=parts[2],
                        queue=parts[3],
                        name=parts[6],
                        cores="-",
                        start_time="-",
                        elapsed="-",
                    ))
            return jobs
    except Exception:
        pass
    return []


def get_finished_jobs_from_files() -> list[Job]:
    """Get finished jobs by scanning output files in the LSF output directory."""
    from src.utils.hpc.jobgen import get_job_output_dir

    output_dir = get_job_output_dir()
    if not output_dir.exists():
        return []

    jobs = []
    for out_file in output_dir.glob("*.out"):
        name = out_file.stem
        # Get file modification time as "finish time"
        mtime = out_file.stat().st_mtime
        time_str = datetime.fromtimestamp(mtime).strftime("%b %d %H:%M")

        jobs.append(Job(
            id="-",
            name=name,
            queue="-",
            status="DONE",
            cores="-",
            start_time=time_str,
            elapsed="-",
            output_file=str(out_file),
        ))

    # Sort by modification time, newest first
    jobs.sort(key=lambda j: Path(j.output_file).stat().st_mtime if j.output_file else 0, reverse=True)
    return jobs


def get_queue_info() -> list[str]:
    """Get cluster queue information."""
    lines = []
    try:
        result = subprocess.run(["bqueues"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines.append("=== Queue Summary ===")
            lines.extend(result.stdout.strip().split("\n"))
    except Exception:
        lines.append("Failed to get queue info")

    try:
        result = subprocess.run(["bhosts", "-w"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines.append("")
            lines.append("=== Host Status ===")
            lines.extend(result.stdout.strip().split("\n"))
    except Exception:
        pass

    return lines


def get_job_info(job_id: str) -> list[str]:
    """Get job metadata (queue, status, etc.) using bjobs -l."""
    lines = []
    try:
        result = subprocess.run(
            ["bjobs", "-l", job_id],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.split("\n"):
                cleaned = " ".join(line.split())
                if cleaned:
                    lines.append(cleaned)
    except Exception as e:
        lines.append(f"Error: {e}")

    return lines if lines else ["No job info available"]


def get_job_output(job_id: str, tail_lines: int = 50) -> list[str]:
    """Get trailing output from job's output file."""
    lines = []
    try:
        peek = subprocess.run(
            ["bjobs", "-o", "output_file", "-noheader", job_id],
            capture_output=True, text=True, timeout=10
        )
        if peek.returncode == 0:
            output_file = peek.stdout.strip()
            if output_file and output_file not in ("-", ""):
                lines.append(f"File: {output_file}")
                lines.append("")
                try:
                    with open(output_file, "r") as f:
                        all_lines = f.readlines()
                        lines.extend([l.rstrip() for l in all_lines[-tail_lines:]])
                except FileNotFoundError:
                    lines.append("(file not yet created)")
                except PermissionError:
                    lines.append("(permission denied)")
                except Exception as e:
                    lines.append(f"(error reading file: {e})")
            else:
                lines.append("(no output file configured)")
        else:
            lines.append(f"(bjobs error: {peek.stderr.strip()})")
    except FileNotFoundError:
        lines.append("(bjobs command not found)")
    except Exception as e:
        lines.append(f"(unable to fetch output: {e})")

    return lines if lines else ["No output available"]


def get_job_details(job_id: str) -> list[str]:
    """Get detailed job information using bjobs -l (legacy combined function)."""
    lines = []
    lines.append("=== Job Details ===")
    lines.append("")
    lines.extend(get_job_info(job_id))
    lines.append("")
    lines.append("=== Output ===")
    lines.append("")
    lines.extend(get_job_output(job_id))
    return lines


def kill_job(job_id: str) -> tuple[bool, str]:
    """Kill a job."""
    try:
        result = subprocess.run(["bkill", job_id], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return False, str(e)


def kill_all_jobs() -> tuple[bool, str]:
    """Kill all jobs."""
    try:
        result = subprocess.run(["bkill", "0"], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return False, str(e)


def draw_floating_window(term, title: str, lines: list[str], scroll: int) -> None:
    """Draw a centered floating window with content."""
    win_width = min(term.width - 4, 100)
    win_height = min(term.height - 6, 30)
    start_x = (term.width - win_width) // 2
    start_y = (term.height - win_height) // 2

    print(term.move_xy(start_x, start_y) + term.cyan + "╭" + "─" * (win_width - 2) + "╮" + term.normal)

    title_text = f" {title} "
    padding = win_width - 4 - len(title_text)
    print(term.move_xy(start_x, start_y + 1) + term.cyan + "│" + term.normal +
          term.bold + title_text + term.normal + " " * padding +
          term.bright_black + "[j/k scroll, Esc close]" + term.normal +
          term.cyan + " │" + term.normal)
    print(term.move_xy(start_x, start_y + 2) + term.cyan + "├" + "─" * (win_width - 2) + "┤" + term.normal)

    content_height = win_height - 4
    visible = lines[scroll:scroll + content_height]

    for i in range(content_height):
        line_content = visible[i][:win_width - 4] if i < len(visible) else ""
        padding = win_width - 4 - len(line_content)
        print(term.move_xy(start_x, start_y + 3 + i) +
              term.cyan + "│ " + term.normal + line_content + " " * padding + term.cyan + " │" + term.normal)

    print(term.move_xy(start_x, start_y + win_height - 1) + term.cyan + "╰" + "─" * (win_width - 2) + "╯" + term.normal)


def run_monitor(term=None):
    """Run the HPC monitor. If term is provided, use it, else create one."""
    if term is None:
        term = Terminal()
        
    selected = 0
    message = ""
    views = ["all", "running", "pending", "resources"]
    view_idx = 0
    output_lines: list[str] = []
    resource_lines: list[str] = []
    scroll = 0
    modal_open = False
    modal_scroll = 0

    all_jobs = get_jobs()

    # If we are passed a terminal, we assume we are already in a context or should create one.
    # But for simplicity, let's just use the context manager here. 
    # Note: If called from another blessed app, nested fullscreen() might be weird, 
    # but let's assume this takes over the screen.
    with term.fullscreen(), term.cbreak(), term.hidden_cursor():
        while True:
            view = views[view_idx]

            if view == "running":
                jobs = [j for j in all_jobs if j.status == "RUN"]
            elif view == "pending":
                jobs = [j for j in all_jobs if j.status == "PEND"]
            else:
                jobs = all_jobs

            print(term.home + term.clear)

            tabs = []
            for i, v in enumerate(views):
                label = v.upper()
                if v == "all":
                    label = f"ALL ({len(all_jobs)})"
                elif v == "running":
                    label = f"RUN ({len([j for j in all_jobs if j.status == 'RUN'])})"
                elif v == "pending":
                    label = f"PEND ({len([j for j in all_jobs if j.status == 'PEND'])})"

                if i == view_idx:
                    tabs.append(term.reverse + f" {label} " + term.normal)
                else:
                    tabs.append(f" {label} ")

            tab_line = " │ ".join(tabs)
            print(term.bold + term.cyan + " HPC Monitor " + term.normal + "  " + tab_line)
            print(term.cyan + "─" * term.width + term.normal)

            if view in ("all", "running", "pending"):
                hdr = f"{ 'ID':<10} {'Name':<30} {'Queue':<6} {'#':<4} {'Status':<6} {'Started':<14} {'Elapsed':<10}"
                print(term.bold + hdr[:term.width] + term.normal)
                print(term.bright_black + "─" * term.width + term.normal)

                max_rows = term.height - 8
                for i, job in enumerate(jobs[:max_rows]):
                    name = job.name[:28] + ".." if len(job.name) > 30 else job.name

                    if job.status == "RUN":
                        status = term.green + f"{job.status:<6}" + term.normal
                    elif job.status == "PEND":
                        status = term.yellow + f"{job.status:<6}" + term.normal
                    else:
                        status = term.bright_black + f"{job.status:<6}" + term.normal

                    row = f"{job.id:<10} {name:<30} {job.queue:<6} {job.cores:<4} {status} {job.start_time:<14} {job.elapsed:<10}"

                    if i == selected:
                        print(term.reverse + row[:term.width] + term.normal)
                    else:
                        print(row[:term.width])

                if not jobs:
                    print(term.bright_black + "  No jobs" + term.normal)

            elif view == "resources":
                max_rows = term.height - 6
                visible = resource_lines[scroll:scroll + max_rows]
                for line in visible:
                    print(line[:term.width])

            print(term.move_y(term.height - 3) + term.cyan + "─" * term.width + term.normal)

            if message:
                print(term.yellow + f" {message}" + term.normal)
            else:
                print()

            help_text = " [h/l] Tabs  [j/k] Navigate  [t] Job details  [d] Kill  [D] Kill All  [r] Refresh  [q] Quit"
            print(term.bright_black + help_text[:term.width] + term.normal)

            if modal_open and output_lines:
                job_name = jobs[selected].name if jobs and 0 <= selected < len(jobs) else "Output"
                draw_floating_window(term, job_name, output_lines, modal_scroll)

            key = term.inkey(timeout=5)

            if key:
                message = ""
                if modal_open:
                    if key.name == 'KEY_ESCAPE' or key == 't' or key == 'q':
                        modal_open = False
                    elif key == 'j' or key.name == 'KEY_DOWN':
                        modal_scroll = min(modal_scroll + 1, max(0, len(output_lines) - 10))
                    elif key == 'k' or key.name == 'KEY_UP':
                        modal_scroll = max(0, modal_scroll - 1)
                    elif key == 'G':
                        modal_scroll = max(0, len(output_lines) - 10)
                    elif key == 'g':
                        modal_scroll = 0
                    elif key.lower() == 'r':
                        if jobs and 0 <= selected < len(jobs):
                            output_lines = get_job_details(jobs[selected].id)
                            message = "Refreshed"
                    continue

                if key.lower() == 'q' or key.name == 'KEY_ESCAPE':
                    return  # Exit monitor, return to main menu

                elif key.lower() == 'r':
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))
                    if view == "resources":
                        resource_lines = get_queue_info()
                    message = "Refreshed"

                elif key == 'h' or key.name == 'KEY_LEFT':
                    view_idx = (view_idx - 1) % len(views)
                    selected = 0
                    scroll = 0
                    if views[view_idx] == "resources":
                        resource_lines = get_queue_info()

                elif key == 'l' or key.name == 'KEY_RIGHT':
                    view_idx = (view_idx + 1) % len(views)
                    selected = 0
                    scroll = 0
                    if views[view_idx] == "resources":
                        resource_lines = get_queue_info()

                elif key == 'j' or key.name == 'KEY_DOWN':
                    if view in ("all", "running", "pending"):
                        selected = min(selected + 1, len(jobs) - 1) if jobs else 0
                    elif view == "resources":
                        scroll += 1

                elif key == 'k' or key.name == 'KEY_UP':
                    if view in ("all", "running", "pending"):
                        selected = max(selected - 1, 0)
                    elif view == "resources":
                        scroll = max(0, scroll - 1)

                elif key == 'd' and jobs and view in ("all", "running", "pending"):
                    job = jobs[selected]
                    ok, msg = kill_job(job.id)
                    message = f"Killed: {job.name}" if ok else f"Failed: {msg}"
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))

                elif key == 'D':
                    ok, msg = kill_all_jobs()
                    message = "Killed all jobs" if ok else f"Failed: {msg}"
                    all_jobs = get_jobs()
                    selected = 0

                elif key == 't' and jobs and view in ("all", "running", "pending"):
                    output_lines = get_job_details(jobs[selected].id)
                    modal_scroll = 0
                    modal_open = True

            else:
                if not modal_open:
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))

if __name__ == "__main__":
    run_monitor()
