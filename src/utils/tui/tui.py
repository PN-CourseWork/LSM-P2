"""Main TUI application using blessed."""

import subprocess
from pathlib import Path

from blessed import Terminal

from src.utils import runners, mlflow
from src.utils.hpc import get_pack_files, get_available_groups, submit_pack
from src.utils.hpc.submit import get_group_config_preview
from src.utils.config import load_project_config, clean_all
from src.utils.tui.monitor import get_jobs, get_finished_jobs_from_files
from src.utils.tui.runner import TuiRunner
from src.utils.tui.actions import docs


class TuiApp:
    def __init__(self):
        self.term = Terminal()
        self.runner = TuiRunner(self.term)
        self.tabs = ["Actions", "HPC"]
        self.current_tab_idx = 0

        # --- Actions Tab State ---
        self.actions = [
            {
                "label": "Run Compute Scripts",
                "category": "Compute",
                "action": self._run_compute,
                "description": [
                    "Executes all compute scripts in the project.",
                    "Runs numerical simulations and data processing.",
                ],
            },
            {
                "label": "Run Plot Scripts",
                "category": "Compute",
                "action": self._run_plots,
                "description": [
                    "Generates all plots from computed data.",
                    "Plots saved to configured output directory.",
                ],
            },
            {
                "label": "Copy Plots to Report",
                "category": "Compute",
                "action": self._copy_plots,
                "description": [
                    "Copies generated plots to the LaTeX report",
                    "figures directory.",
                ],
            },
            {
                "label": "Fetch MLflow Artifacts",
                "category": "Data",
                "action": self._fetch_mlflow,
                "description": [
                    "Downloads experiment artifacts from the",
                    "configured MLflow tracking server.",
                ],
            },
            {
                "label": "Build Documentation",
                "category": "Docs",
                "action": self._build_docs,
                "description": [
                    "Builds project documentation using Sphinx.",
                    "Output generated in docs/_build/html/",
                ],
            },
            {
                "label": "Clean Generated Files",
                "category": "Maintenance",
                "action": self._clean_all,
                "description": [
                    "Removes all generated files including:",
                    "computed data, plots, and build artifacts.",
                ],
            },
        ]
        self.action_idx = 0

        # --- Global Output State (shared terminal output pane) ---
        self.output_lines = []  # Persistent terminal output
        self.output_scroll = 0

        # --- HPC Tab State ---
        self.hpc_cmd_idx = 0
        self.hpc_job_idx = 0
        self.hpc_jobs = []  # All jobs
        self.hpc_jobs_active = []  # Running jobs
        self.hpc_jobs_pending = []  # Pending jobs
        self.hpc_jobs_finished = []  # Done/Exit jobs
        self.hpc_job_tab = 0  # 0=Active, 1=Pending, 2=Finished

        # Mode: "commands" | "generate_select" | "submit_select"
        self.hpc_mode = "commands"
        self.hpc_select_items = []  # Items for selection modes
        self.hpc_select_idx = 0
        self.hpc_selected = set()  # For multi-select

        # Focus: "commands" | "jobs" - which pane has focus
        self.hpc_focus = "commands"

        # HPC Commands with descriptions
        self.hpc_commands = [
            {
                "label": "Generate Job Pack",
                "action": self.enter_generate_mode,
                "description": [
                    "Creates LSF job pack files from packs.yaml.",
                    "Select groups, generate with parameter sweeps.",
                ],
            },
            {
                "label": "Submit Job Pack",
                "action": self.enter_submit_mode,
                "description": [
                    "Submit a generated .pack file to LSF.",
                    "Requires bsub command and LSF environment.",
                ],
            },
        ]

    def log(self, *lines):
        """Append lines to the persistent output pane."""
        for line in lines:
            if isinstance(line, list):
                self.output_lines.extend(line)
            else:
                self.output_lines.append(str(line))
        # Auto-scroll to bottom
        self.output_scroll = max(0, len(self.output_lines) - 10)

    def log_header(self, title: str):
        """Add a header line to output."""
        self.log("", f"─── {title} ───")

    def get_project_paths(self):
        repo_root = Path.cwd()
        project_config = load_project_config()
        hpc_config = project_config.get("hpc", {})
        job_packs_dir_str = hpc_config.get("job_packs", "Experiments/05-scaling/job-packs")
        job_packs_dir = repo_root / job_packs_dir_str
        config_path = job_packs_dir / "packs.yaml"
        return config_path, job_packs_dir, repo_root, project_config

    # --- HPC Mode Methods ---

    def enter_generate_mode(self):
        """Enter generate mode - show group selection in commands pane."""
        config_path, _, _, _ = self.get_project_paths()
        groups = get_available_groups(config_path)

        if not groups:
            self.runner.show_output("Generate Job Pack", ["No groups found in packs.yaml"])
            return

        self.hpc_mode = "generate_select"
        self.hpc_select_items = groups
        self.hpc_select_idx = 0
        self.hpc_selected = set()

    def enter_submit_mode(self):
        """Enter submit mode - show pack file selection in commands pane."""
        _, job_packs_dir, _, _ = self.get_project_paths()
        pack_files = get_pack_files(job_packs_dir)

        if not pack_files:
            self.log_header("Submit Job Pack")
            self.log(f"No .pack files found in {job_packs_dir}")
            return

        self.hpc_mode = "submit_select"
        self.hpc_select_items = pack_files
        self.hpc_select_idx = 0

    def exit_select_mode(self):
        """Exit selection mode back to commands."""
        self.hpc_mode = "commands"
        self.hpc_select_items = []
        self.hpc_select_idx = 0
        self.hpc_selected = set()

    def execute_generate(self):
        """Execute generation with selected groups."""
        if not self.hpc_selected:
            return

        config_path, job_packs_dir, _, _ = self.get_project_paths()
        selected_groups = [self.hpc_select_items[i] for i in sorted(self.hpc_selected)]

        self.log_header("Generate Job Pack")
        self.log(f"Groups: {', '.join(selected_groups)}")

        # Import and run generation
        from src.utils.hpc.submit import generate_pack
        files, log = generate_pack(config_path, job_packs_dir, selected_groups)

        # Log output
        self.log(log)
        if files:
            self.log(f"✓ Generated {len(files)} pack file(s)")
        else:
            self.log("✗ No files generated")

        self.exit_select_mode()

    def execute_submit(self):
        """Execute submission of selected pack file."""
        if self.hpc_select_idx >= len(self.hpc_select_items):
            return

        selected_file = self.hpc_select_items[self.hpc_select_idx]

        # Count jobs
        with open(selected_file, "r") as f:
            job_count = len([l for l in f if l.strip() and not l.startswith('#')])

        # Confirm
        if not self.runner.confirm(f"Submit {selected_file.name} ({job_count} jobs) to LSF?"):
            self.exit_select_mode()
            return

        self.log_header("Submit Job Pack")
        self.log(f"File: {selected_file.name} ({job_count} jobs)")

        # Submit
        success, log = submit_pack(selected_file)

        # Log output
        self.log(log)
        if success:
            self.log("✓ Submission successful")
        else:
            self.log("✗ Submission failed")

        self.exit_select_mode()

    def _build_pack_preview(self, pack_file: Path) -> list[str]:
        """Build preview lines for a pack file."""
        _, job_packs_dir, _, _ = self.get_project_paths()
        preview = []

        # Get group name from pack file name
        group_name = pack_file.stem

        # Show YAML group config
        config_path = job_packs_dir / "packs.yaml"
        if config_path.exists():
            preview.extend(get_group_config_preview(config_path, group_name))
            preview.append("")
            preview.append("─" * 50)
            preview.append("")

        # Show pack file preview
        with open(pack_file, "r") as f:
            pack_lines = f.readlines()

        job_count = len([l for l in pack_lines if l.strip() and not l.startswith('#')])

        preview.append(f"Pack File: {pack_file.name}")
        preview.append(f"Total Jobs: {job_count}")
        preview.append("")
        preview.append("Generated Commands (first 5):")
        preview.append("-" * 30)

        shown = 0
        for line in pack_lines:
            line = line.rstrip()
            if line and not line.startswith('#'):
                if len(line) > 80:
                    preview.append(f"  {line[:77]}...")
                else:
                    preview.append(f"  {line}")
                shown += 1
                if shown >= 5:
                    break

        if job_count > 5:
            preview.append(f"  ... ({job_count - 5} more jobs)")

        return preview

    # --- Action execution methods ---

    def _run_action(self, name: str, func):
        """Run an action and log output."""
        self.log_header(name)
        self.log("Running...")
        self.draw()

        result = self.runner.capture_output(func)
        self.log(result.output)
        if result.error:
            self.log(f"Error: {result.error}")
        self.log("✓ Done" if result.success else "✗ Failed")

    def _run_compute(self):
        """Run compute scripts."""
        self._run_action("Run Compute Scripts", runners.run_compute_scripts)

    def _run_plots(self):
        """Run plot scripts."""
        self._run_action("Run Plot Scripts", runners.run_plot_scripts)

    def _copy_plots(self):
        """Copy plots to report."""
        self._run_action("Copy Plots to Report", runners.copy_to_report)

    def _fetch_mlflow(self):
        """Fetch MLflow artifacts."""
        _, _, repo_root, config = self.get_project_paths()
        mlflow_conf = config.get("mlflow", {})

        def _fetch():
            mlflow.setup_mlflow_tracking()
            output_dir = repo_root / mlflow_conf.get("download_dir", "data")
            mlflow.fetch_project_artifacts(output_dir)

        self._run_action("Fetch MLflow Artifacts", _fetch)

    def _build_docs(self):
        """Build documentation and open in browser."""
        def build_and_open():
            success = docs.build_docs()
            if success:
                docs_index = Path.cwd() / "docs" / "build" / "html" / "index.html"
                if docs_index.exists():
                    subprocess.run(["xdg-open", str(docs_index)])
                    print("Opened documentation in browser")
                else:
                    print(f"Docs not found at: {docs_index}")

        self._run_action("Build Documentation", build_and_open)

    def _clean_all(self):
        """Clean generated files with confirmation."""
        if self.runner.confirm("Clean all generated files?"):
            self._run_action("Clean Generated Files", clean_all)

    def refresh_jobs(self):
        """Fetch jobs for the HPC jobs pane and categorize them."""
        self.hpc_jobs = get_jobs()
        self.hpc_jobs_active = [j for j in self.hpc_jobs if j.status == "RUN"]
        self.hpc_jobs_pending = [j for j in self.hpc_jobs if j.status == "PEND"]

        # Get finished jobs from LSF + from output files
        lsf_finished = [j for j in self.hpc_jobs if j.status not in ("RUN", "PEND")]
        file_finished = get_finished_jobs_from_files()

        # Merge: avoid duplicates by name
        seen_names = {j.name for j in lsf_finished}
        self.hpc_jobs_finished = lsf_finished + [j for j in file_finished if j.name not in seen_names]

    def get_current_job_list(self):
        """Get the job list for the current tab."""
        if self.hpc_job_tab == 0:
            return self.hpc_jobs_active
        elif self.hpc_job_tab == 1:
            return self.hpc_jobs_pending
        else:
            return self.hpc_jobs_finished

    def get_hpc_selection_preview(self) -> list[str]:
        """Get preview content for HPC selection modes."""
        config_path, job_packs_dir, _, _ = self.get_project_paths()

        if self.hpc_mode == "generate_select":
            # Show config for selected/hovered group
            if self.hpc_select_items and self.hpc_select_idx < len(self.hpc_select_items):
                group_name = self.hpc_select_items[self.hpc_select_idx]
                lines = get_group_config_preview(config_path, group_name)
                if self.hpc_selected:
                    lines.append("")
                    lines.append("─" * 40)
                    lines.append(f"Selected: {len(self.hpc_selected)} group(s)")
                    for idx in sorted(self.hpc_selected):
                        lines.append(f"  • {self.hpc_select_items[idx]}")
                return lines
            return ["No groups available"]

        elif self.hpc_mode == "submit_select":
            # Show pack preview for selected file
            if self.hpc_select_items and self.hpc_select_idx < len(self.hpc_select_items):
                pack_file = self.hpc_select_items[self.hpc_select_idx]
                return self._build_pack_preview(pack_file)
            return ["No pack files available"]

        return []

    def get_job_output_lines(self, job) -> list[str]:
        """Get the tail of a job's output file."""
        # If job has output_file set (from file scan), read it directly
        if job.output_file:
            try:
                with open(job.output_file, "r") as f:
                    lines = f.readlines()
                    return [l.rstrip() for l in lines[-100:]]
            except Exception as e:
                return [f"Error reading {job.output_file}: {e}"]

        # Otherwise use bjobs to get output file path
        from src.utils.tui.monitor import get_job_output
        return get_job_output(job.id, tail_lines=100)

    # --- Drawing Methods ---

    def _draw_job_output_pane(self, x: int, y: int, width: int, height: int):
        """Draw the job output pane showing tail of selected job's output file."""
        term = self.term
        jobs = self.get_current_job_list()

        if not jobs or self.hpc_job_idx >= len(jobs):
            print(term.move_xy(x, y) + term.bold(" Job Output "))
            print(term.move_xy(x, y + 2) + term.bright_black("(no job selected)"))
            return

        job = jobs[self.hpc_job_idx]
        lines = self.get_job_output_lines(job)

        # Title with job name
        title = f" {job.name} "
        if len(title) > width:
            title = title[:width-3] + "..."
        print(term.move_xy(x, y) + term.bold(title))

        if not lines:
            print(term.move_xy(x, y + 2) + term.bright_black("(no output yet)"))
            return

        # Show last lines that fit in the pane (auto-scroll to bottom)
        visible_height = height - 2
        visible = lines[-visible_height:] if len(lines) > visible_height else lines

        for i, line in enumerate(visible):
            line_y = y + 2 + i
            disp = line[:width] if len(line) > width else line
            print(term.move_xy(x, line_y) + disp)

        # Line count indicator
        info = f"[{len(lines)} lines]"
        print(term.move_xy(x + width - len(info), y) + term.bright_black(info))

    def _draw_output_pane(self, x: int, y: int, width: int, height: int):
        """Draw the persistent output pane on the right."""
        term = self.term

        # Title
        pane_title = " Output "
        print(term.move_xy(x, y) + term.bold(pane_title))

        if not self.output_lines:
            print(term.move_xy(x, y + 2) + term.bright_black("(no output yet)"))
            print(term.move_xy(x, y + 3) + term.bright_black("Run an action to see output"))
            return

        # Content
        visible_height = height - 2
        visible = self.output_lines[self.output_scroll:self.output_scroll + visible_height]

        for i, line in enumerate(visible):
            line_y = y + 2 + i
            disp = line[:width] if len(line) > width else line
            print(term.move_xy(x, line_y) + disp)

        # Scroll indicator
        if len(self.output_lines) > visible_height:
            max_scroll = max(1, len(self.output_lines) - visible_height)
            pct = int((self.output_scroll / max_scroll) * 100)
            info = f"[{pct}%] {self.output_scroll + 1}-{min(self.output_scroll + visible_height, len(self.output_lines))}/{len(self.output_lines)}"
            print(term.move_xy(x + width - len(info), y + height - 1) + term.bright_black(info))

    def draw_actions_tab(self):
        """Draw Actions tab: Left (Description top, Actions bottom), Right (Output)."""
        term = self.term
        content_start_y = 4
        content_end_y = term.height - 2

        # Left/Right split (40% left, 60% right for output)
        left_width = int(term.width * 0.4)
        sep_x = left_width

        # Left pane split: evenly between description and actions
        left_split_y = content_start_y + (content_end_y - content_start_y) // 2

        # Draw separators
        for y in range(content_start_y, content_end_y + 1):
            print(term.move_xy(sep_x, y) + term.cyan("│"))
        print(term.move_xy(0, left_split_y) + term.cyan("─" * left_width + "┤"))

        # === Top-Left: Description ===
        action = self.actions[self.action_idx]
        print(term.move_xy(2, content_start_y) + term.bold(f" {action['label']} "))

        for i, line in enumerate(action["description"]):
            y = content_start_y + 2 + i
            if y >= left_split_y:
                break
            disp = line[:left_width - 3] if len(line) > left_width - 3 else line
            print(term.move_xy(2, y) + term.bright_black(disp))

        # === Bottom-Left: Actions List ===
        print(term.move_xy(2, left_split_y + 1) + term.bold(" Actions "))

        for i, act in enumerate(self.actions):
            y = left_split_y + 3 + i
            if y >= content_end_y:
                break
            is_selected = i == self.action_idx
            prefix = ">" if is_selected else " "
            style = term.bold if is_selected else term.normal
            cat = act["category"]
            max_label = left_width - len(cat) - 7
            label = act["label"][:max_label] if len(act["label"]) > max_label else act["label"]
            print(term.move_xy(1, y) + f"{prefix} {style}{label}{term.normal} {term.bright_black}[{cat}]{term.normal}")

        # === Right: Output Pane ===
        output_width = term.width - sep_x - 3
        output_height = content_end_y - content_start_y
        self._draw_output_pane(sep_x + 2, content_start_y, output_width, output_height)

    def draw_hpc_tab(self):
        """Draw HPC tab: Left (Preview/Commands top, Jobs bottom), Right (Output)."""
        term = self.term
        content_start_y = 4
        content_end_y = term.height - 2

        # Left/Right split (40% left, 60% right for output)
        left_width = int(term.width * 0.4)
        sep_x = left_width

        # Left pane split: evenly between top and bottom
        left_split_y = content_start_y + (content_end_y - content_start_y) // 2

        # Draw separators
        for y in range(content_start_y, content_end_y + 1):
            print(term.move_xy(sep_x, y) + term.cyan("│"))
        print(term.move_xy(0, left_split_y) + term.cyan("─" * left_width + "┤"))

        # === Top-Left: Preview/Commands or Selection ===
        if self.hpc_mode == "generate_select":
            self._draw_generate_select(content_start_y, left_split_y, left_width)
        elif self.hpc_mode == "submit_select":
            self._draw_submit_select(content_start_y, left_split_y, left_width)
        else:
            self._draw_hpc_preview_and_commands(content_start_y, left_split_y, left_width)

        # === Bottom-Left: Jobs ===
        self._draw_jobs_pane(left_split_y + 1, content_end_y, left_width)

        # === Right: Output or Selection Preview ===
        output_width = term.width - sep_x - 3
        output_height = content_end_y - content_start_y

        if self.hpc_mode in ("generate_select", "submit_select"):
            # Show selection preview in right pane
            self._draw_selection_preview(sep_x + 2, content_start_y, output_width, output_height)
        elif self.hpc_focus == "jobs" and self.get_current_job_list():
            # Show job output when focused on jobs pane
            self._draw_job_output_pane(sep_x + 2, content_start_y, output_width, output_height)
        else:
            self._draw_output_pane(sep_x + 2, content_start_y, output_width, output_height)

    def _draw_selection_preview(self, x: int, y: int, width: int, height: int):
        """Draw preview for selection mode in right pane."""
        term = self.term
        print(term.move_xy(x, y) + term.bold(" Preview "))

        lines = self.get_hpc_selection_preview()
        if not lines:
            print(term.move_xy(x, y + 2) + term.bright_black("(no preview)"))
            return

        visible = lines[:height - 2]
        for i, line in enumerate(visible):
            line_y = y + 2 + i
            disp = line[:width] if len(line) > width else line
            print(term.move_xy(x, line_y) + disp)

    def _draw_hpc_preview_and_commands(self, start_y: int, split_y: int, width: int):
        """Draw the preview and commands in top-left of HPC tab."""
        term = self.term
        has_focus = self.hpc_focus == "commands"

        # Show description of selected command
        cmd = self.hpc_commands[self.hpc_cmd_idx]
        print(term.move_xy(2, start_y) + term.bold(f" {cmd['label']} "))

        for i, line in enumerate(cmd["description"]):
            y = start_y + 2 + i
            desc_height = split_y - start_y - len(self.hpc_commands) - 4
            if i >= desc_height:
                break
            disp = line[:width - 3] if len(line) > width - 3 else line
            print(term.move_xy(2, y) + term.bright_black(disp))

        # Commands list at bottom of top section
        cmds_start_y = split_y - len(self.hpc_commands) - 2
        print(term.move_xy(0, cmds_start_y) + term.cyan("─" * width + "┤"))

        # Title with focus indicator
        title = " Commands "
        if has_focus:
            print(term.move_xy(2, cmds_start_y + 1) + term.reverse(title))
        else:
            print(term.move_xy(2, cmds_start_y + 1) + term.bold(title))

        for i, c in enumerate(self.hpc_commands):
            y = cmds_start_y + 2 + i
            is_selected = i == self.hpc_cmd_idx and has_focus
            prefix = ">" if is_selected else " "
            style = term.bold if is_selected else term.normal
            max_label = width - 4
            label = c["label"][:max_label] if len(c["label"]) > max_label else c["label"]
            print(term.move_xy(1, y) + f"{prefix} {style}{label}{term.normal}")

    def _draw_generate_select(self, start_y: int, split_y: int, width: int):
        """Draw generate group selection pane."""
        term = self.term
        print(term.move_xy(1, start_y) + term.reverse(" Select Groups [Space=toggle, Enter=gen, Esc=cancel] "[:width-2]))

        max_rows = split_y - start_y - 2
        start_idx = max(0, self.hpc_select_idx - max_rows + 1)
        visible = self.hpc_select_items[start_idx:start_idx + max_rows]

        for i, group in enumerate(visible):
            y = start_y + 2 + i
            real_idx = start_idx + i
            is_current = real_idx == self.hpc_select_idx
            is_checked = real_idx in self.hpc_selected

            checkbox = "[x]" if is_checked else "[ ]"
            prefix = ">" if is_current else " "
            style = term.bold if is_current else term.normal

            max_label = width - 8
            disp = group[:max_label] if len(group) > max_label else group
            print(term.move_xy(1, y) + f"{prefix} {checkbox} {style}{disp}{term.normal}")

    def _draw_submit_select(self, start_y: int, split_y: int, width: int):
        """Draw submit pack file selection pane."""
        term = self.term
        print(term.move_xy(1, start_y) + term.reverse(" Select Pack [Enter=submit, Esc=cancel] "[:width-2]))

        max_rows = split_y - start_y - 2
        start_idx = max(0, self.hpc_select_idx - max_rows + 1)
        visible = self.hpc_select_items[start_idx:start_idx + max_rows]

        for i, pack_file in enumerate(visible):
            y = start_y + 2 + i
            real_idx = start_idx + i
            is_current = real_idx == self.hpc_select_idx

            prefix = ">" if is_current else " "
            style = term.bold if is_current else term.normal

            max_label = width - 4
            name = pack_file.name
            disp = name[:max_label] if len(name) > max_label else name
            print(term.move_xy(1, y) + f"{prefix} {style}{disp}{term.normal}")

    def _draw_jobs_pane(self, start_y: int, end_y: int, width: int):
        """Draw the jobs list pane with tabs."""
        term = self.term
        has_focus = self.hpc_focus == "jobs"

        # Title with focus indicator
        title = " Jobs "
        if has_focus:
            print(term.move_xy(1, start_y) + term.reverse(title))
        else:
            print(term.move_xy(1, start_y) + term.bold(title))

        # Draw job tabs: Active | Pending | Finished
        tab_names = [
            f"Active({len(self.hpc_jobs_active)})",
            f"Pending({len(self.hpc_jobs_pending)})",
            f"Done({len(self.hpc_jobs_finished)})",
        ]

        tab_x = 1
        tab_y = start_y + 1
        for i, name in enumerate(tab_names):
            if i == self.hpc_job_tab:
                style = term.reverse if has_focus else term.underline
            else:
                style = term.normal
            print(term.move_xy(tab_x, tab_y) + f"{style} {name} {term.normal}")
            tab_x += len(name) + 3

        # Get jobs for current tab
        jobs = self.get_current_job_list()

        # Job list header
        header_y = start_y + 2
        print(term.move_xy(2, header_y) + term.bright_black(f"{'ID':<8} {'Name'}"))

        # Job list
        job_start_y = header_y + 1
        max_job_rows = end_y - job_start_y

        if jobs:
            start_idx = max(0, self.hpc_job_idx - max_job_rows + 1)
            visible_jobs = jobs[start_idx:start_idx + max_job_rows]

            for i, job in enumerate(visible_jobs):
                y = job_start_y + i
                if y >= end_y:
                    break
                real_idx = start_idx + i
                is_selected = real_idx == self.hpc_job_idx and has_focus
                prefix = ">" if is_selected else " "
                style = term.bold if is_selected else term.normal

                max_name_len = width - 12
                name = job.name[:max_name_len] if len(job.name) > max_name_len else job.name
                line = f"{prefix}{job.id:<8} {style}{name}{term.normal}"
                print(term.move_xy(1, y) + line)
        else:
            tab_label = ["active", "pending", "finished"][self.hpc_job_tab]
            print(term.move_xy(3, job_start_y) + term.bright_black(f"(No {tab_label} jobs)"))
            print(term.move_xy(3, job_start_y + 1) + term.bright_black("[1/2/3] Switch tabs"))

    def draw(self):
        """Main draw method."""
        print(self.term.home + self.term.clear)

        # Header
        header = f" {self.term.bold}Project Manager{self.term.normal} "
        print(self.term.black_on_white(header.center(self.term.width)))
        print()

        # Tabs
        tab_line = ""
        for i, tab in enumerate(self.tabs):
            if i == self.current_tab_idx:
                tab_line += f" {self.term.reverse} {tab} {self.term.normal} "
            else:
                tab_line += f"  {tab}  "
        print(tab_line.center(self.term.width))
        print(self.term.cyan("─" * self.term.width))

        current_tab = self.tabs[self.current_tab_idx]

        if current_tab == "Actions":
            self.draw_actions_tab()
        elif current_tab == "HPC":
            self.draw_hpc_tab()

        # Status Line
        if current_tab == "HPC":
            if self.hpc_focus == "jobs":
                help_msg = " [Tab] Switch pane | [j/k] Nav | [1/2/3] Tabs | [x] Kill | [J/K] Scroll | [r] Refresh | [q] Quit"
            else:
                help_msg = " [Tab] Switch pane | [j/k] Nav | [1/2/3] Job tabs | [Enter] Run | [J/K] Scroll | [r] Refresh | [q] Quit"
        else:
            help_msg = " [j/k] Nav | [Enter] Run | [J/K] Scroll | [r] Refresh | [c] Clear | [G] Git | [q] Quit"

        status_bar = f"{help_msg:<{self.term.width}}"
        print(
            self.term.move_xy(0, self.term.height - 1) + self.term.black_on_white(status_bar),
            end="",
            flush=True,
        )

    def run(self):
        """Main TUI Loop."""
        self.refresh_jobs()

        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            while True:
                self.draw()
                key = self.term.inkey(timeout=5)

                # Auto-refresh jobs on timeout
                if not key:
                    if self.tabs[self.current_tab_idx] == "HPC":
                        self.refresh_jobs()
                    continue

                # --- Global Keys ---
                if key.lower() == "q" and self.hpc_mode == "commands":
                    break

                if key == "G":
                    subprocess.run(["lazygit"])
                    continue

                if key.lower() == "r":
                    self.refresh_jobs()
                    continue

                if key == "c":
                    # Clear output
                    self.output_lines = []
                    self.output_scroll = 0
                    continue

                # Shift+J/K to scroll output
                if key == "J":
                    max_scroll = max(0, len(self.output_lines) - 10)
                    self.output_scroll = min(max_scroll, self.output_scroll + 1)
                    continue
                if key == "K":
                    self.output_scroll = max(0, self.output_scroll - 1)
                    continue

                # Tab navigation with h/l
                if key == "h":
                    self.current_tab_idx = (self.current_tab_idx - 1) % len(self.tabs)
                    continue
                if key == "l":
                    self.current_tab_idx = (self.current_tab_idx + 1) % len(self.tabs)
                    continue

                # --- Actions Tab ---
                if self.tabs[self.current_tab_idx] == "Actions":
                    if key.name == "KEY_UP" or key == "k":
                        self.action_idx = max(0, self.action_idx - 1)
                    elif key.name == "KEY_DOWN" or key == "j":
                        self.action_idx = min(len(self.actions) - 1, self.action_idx + 1)
                    elif key.name == "KEY_ENTER":
                        self.actions[self.action_idx]["action"]()

                # --- HPC Tab ---
                elif self.tabs[self.current_tab_idx] == "HPC":
                    # Selection mode: generate
                    if self.hpc_mode == "generate_select":
                        if key.name == "KEY_ESCAPE":
                            self.exit_select_mode()
                        elif key.name == "KEY_UP" or key == "k":
                            self.hpc_select_idx = max(0, self.hpc_select_idx - 1)
                        elif key.name == "KEY_DOWN" or key == "j":
                            self.hpc_select_idx = min(len(self.hpc_select_items) - 1, self.hpc_select_idx + 1)
                        elif key == " ":
                            if self.hpc_select_idx in self.hpc_selected:
                                self.hpc_selected.remove(self.hpc_select_idx)
                            else:
                                self.hpc_selected.add(self.hpc_select_idx)
                        elif key.name == "KEY_ENTER":
                            if not self.hpc_selected:
                                self.hpc_selected.add(self.hpc_select_idx)
                            self.execute_generate()

                    # Selection mode: submit
                    elif self.hpc_mode == "submit_select":
                        if key.name == "KEY_ESCAPE":
                            self.exit_select_mode()
                        elif key.name == "KEY_UP" or key == "k":
                            self.hpc_select_idx = max(0, self.hpc_select_idx - 1)
                        elif key.name == "KEY_DOWN" or key == "j":
                            self.hpc_select_idx = min(len(self.hpc_select_items) - 1, self.hpc_select_idx + 1)
                        elif key.name == "KEY_ENTER":
                            self.execute_submit()

                    # Normal commands mode
                    else:
                        # TAB to switch focus between commands and jobs
                        if key.name == "KEY_TAB":
                            self.hpc_focus = "jobs" if self.hpc_focus == "commands" else "commands"

                        # Job tab switching with 1/2/3 (works in both focus modes)
                        elif key == "1":
                            self.hpc_job_tab = 0
                            self.hpc_job_idx = 0
                        elif key == "2":
                            self.hpc_job_tab = 1
                            self.hpc_job_idx = 0
                        elif key == "3":
                            self.hpc_job_tab = 2
                            self.hpc_job_idx = 0

                        # Focus-specific navigation
                        elif self.hpc_focus == "commands":
                            if key.name == "KEY_UP" or key == "k":
                                self.hpc_cmd_idx = max(0, self.hpc_cmd_idx - 1)
                            elif key.name == "KEY_DOWN" or key == "j":
                                self.hpc_cmd_idx = min(len(self.hpc_commands) - 1, self.hpc_cmd_idx + 1)
                            elif key.name == "KEY_ENTER":
                                self.hpc_commands[self.hpc_cmd_idx]["action"]()

                        elif self.hpc_focus == "jobs":
                            jobs = self.get_current_job_list()
                            if key.name == "KEY_UP" or key == "k":
                                self.hpc_job_idx = max(0, self.hpc_job_idx - 1)
                            elif key.name == "KEY_DOWN" or key == "j":
                                self.hpc_job_idx = min(len(jobs) - 1, self.hpc_job_idx + 1) if jobs else 0
                            elif key.name == "KEY_ENTER" and jobs:
                                # Show job details in output pane
                                job = jobs[self.hpc_job_idx]
                                self.log_header(f"Job: {job.name}")
                                self.log(f"ID: {job.id}")
                                self.log(f"Status: {job.status}")
                                self.log(f"Queue: {job.queue}")
                            elif key == "x" and jobs:
                                # Kill job
                                job = jobs[self.hpc_job_idx]
                                if self.runner.confirm(f"Kill job {job.id} ({job.name})?"):
                                    self.log_header(f"Killing job {job.id}")
                                    import subprocess
                                    result = subprocess.run(
                                        ["bkill", job.id],
                                        capture_output=True,
                                        text=True,
                                    )
                                    if result.returncode == 0:
                                        self.log("Job killed successfully")
                                        self.log(result.stdout)
                                    else:
                                        self.log("Failed to kill job")
                                        self.log(result.stderr)
                                    self.refresh_jobs()


def run_tui():
    """Entry point to run the TUI application."""
    app = TuiApp()
    app.run()
