"""Main TUI application using blessed."""

import os
import subprocess

import questionary
from blessed import Terminal

from src.utils import runners, mlflow
from src.utils.hpc import interactive_generate, interactive_submit
from src.utils.config import load_project_config, clean_all
from src.utils.tui.monitor import get_jobs, get_job_info, get_job_output
from src.utils.tui.actions import docs
from src.utils.tui.styles import get_custom_style
from pathlib import Path


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


class TuiApp:
    def __init__(self):
        self.term = Terminal()
        self.tabs = ["Actions", "HPC"]
        self.current_tab_idx = 0

        # --- Actions Tab State ---
        self.action_menus = {
            "Compute & Plotting": [
                ("Run Compute Scripts", lambda: runners.run_compute_scripts()),
                ("Run Plot Scripts", lambda: runners.run_plot_scripts()),
                ("Copy Plots to Report", lambda: runners.copy_to_report()),
            ],
            "Data & MLflow": [
                ("Fetch Artifacts", self.action_fetch_mlflow()),
            ],
            "Documentation": [
                ("Build Docs", lambda: docs.build_docs()),
            ],
            "Maintenance": [
                ("Clean Generated Files", self.action_clean_all()),
            ],
        }
        self.action_cat_idx = 0

        # --- HPC Tab State ---
        # Panes: 0=Commands, 1=Jobs, 2=Details, 3=Output
        self.hpc_pane_focus = 0
        self.hpc_cmd_idx = 0
        self.hpc_job_idx = 0
        self.hpc_jobs = []

        # Separate state for Details and Output subpanes
        self.hpc_details_lines = []
        self.hpc_details_scroll = 0
        self.hpc_output_lines = []
        self.hpc_output_scroll = 0

        self.hpc_commands = [
            ("Generate Job Pack", self.action_generate_pack),
            ("Submit Job Pack", self.action_submit_pack),
        ]

    def get_project_paths(self):
        repo_root = Path.cwd()
        project_config = load_project_config()
        hpc_config = project_config.get("hpc", {})
        job_packs_dir_str = hpc_config.get("job_packs", "Experiments/05-scaling/job-packs")
        job_packs_dir = repo_root / job_packs_dir_str
        config_path = job_packs_dir / "packs.yaml"
        return config_path, job_packs_dir, repo_root, project_config

    # --- Action Factories ---

    def action_generate_pack(self):
        config_path, job_packs_dir, _, _ = self.get_project_paths()
        return lambda: interactive_generate(config_path, job_packs_dir)

    def action_submit_pack(self):
        _, job_packs_dir, _, _ = self.get_project_paths()
        return lambda: interactive_submit(job_packs_dir)

    def action_fetch_mlflow(self):
        def _fetch():
            _, _, repo_root, config = self.get_project_paths()
            mlflow_conf = config.get("mlflow", {})
            mlflow.setup_mlflow_auth(mlflow_conf.get("tracking_uri"))
            output_dir = repo_root / mlflow_conf.get("download_dir", "data/downloaded")
            experiments = mlflow_conf.get("experiments", [])
            if not experiments:
                print("  (No experiments configured)")
            else:
                mlflow.fetch_project_artifacts(experiments, output_dir)

        return _fetch

    def action_clean_all(self):
        def _clean():
            if questionary.confirm("Are you sure?", style=get_custom_style()).ask():
                clean_all()

        return _clean

    def refresh_jobs(self):
        """Fetch jobs for the HPC jobs pane."""
        self.hpc_jobs = get_jobs()
        # Update preview if we have a selected job
        if self.hpc_jobs and self.hpc_job_idx < len(self.hpc_jobs):
            self.refresh_job_data()

    def refresh_job_data(self):
        """Fetch details and output for the selected job."""
        if self.hpc_jobs and self.hpc_job_idx < len(self.hpc_jobs):
            job = self.hpc_jobs[self.hpc_job_idx]
            self.hpc_details_lines = get_job_info(job.id)
            self.hpc_output_lines = get_job_output(job.id)
        else:
            self.hpc_details_lines = ["No job selected"]
            self.hpc_output_lines = []

    # --- Drawing Methods ---

    def draw_actions_tab(self):
        """Draw the Actions tab with categories on left, items on right."""
        term = self.term
        categories = list(self.action_menus.keys())

        content_start_y = 4
        mid_x = int(term.width * 0.4)

        # Draw vertical separator
        for y in range(content_start_y, term.height - 1):
            print(term.move_xy(mid_x, y) + term.cyan("│"))

        # Left side: Categories
        print(term.move_xy(2, content_start_y) + term.bold_underline("Categories"))
        for i, cat in enumerate(categories):
            y = content_start_y + 2 + i * 2
            prefix = " > " if i == self.action_cat_idx else "   "
            style = term.bold if i == self.action_cat_idx else term.normal
            print(term.move_xy(0, y) + f"{prefix}{style}{cat}{term.normal}")

        # Right side: Actions for selected category
        cat_name = categories[self.action_cat_idx]
        items = self.action_menus[cat_name]

        print(term.move_xy(mid_x + 2, content_start_y) + term.bold_underline(f"{cat_name}"))
        for i, (label, _) in enumerate(items):
            print(term.move_xy(mid_x + 4, content_start_y + 2 + i) + f"• {label}")

        print(
            term.move_xy(mid_x + 2, content_start_y + 2 + len(items) + 2)
            + term.bright_black("[Enter] Run selected category")
        )

    def draw_hpc_tab(self):
        """Draw HPC tab: Left (Commands/Jobs), Right (Details/Output)."""
        term = self.term
        content_start_y = 4
        content_end_y = term.height - 2

        # Left/Right split
        left_width = int(term.width * 0.35)
        sep_x = left_width

        # Left pane: Commands on top, Jobs on bottom
        cmd_height = len(self.hpc_commands) + 4
        left_split_y = content_start_y + cmd_height

        # Right pane: Details on top, Output on bottom (50/50 split)
        right_height = content_end_y - content_start_y
        right_split_y = content_start_y + (right_height // 3)  # Details gets 1/3, Output gets 2/3

        # Draw vertical separator (left | right)
        for y in range(content_start_y, content_end_y + 1):
            print(term.move_xy(sep_x, y) + term.cyan("│"))

        # Draw horizontal separator on left (commands / jobs)
        print(term.move_xy(0, left_split_y) + term.cyan("─" * left_width + "┤"))

        # Draw horizontal separator on right (details / output)
        print(term.move_xy(sep_x, right_split_y) + term.cyan("├" + "─" * (term.width - sep_x - 1)))

        # === Top-Left: Commands ===
        pane_title = " Commands "
        if self.hpc_pane_focus == 0:
            print(term.move_xy(2, content_start_y) + term.reverse(pane_title))
        else:
            print(term.move_xy(2, content_start_y) + term.bold(pane_title))

        for i, (label, _) in enumerate(self.hpc_commands):
            y = content_start_y + 2 + i
            is_selected = self.hpc_pane_focus == 0 and i == self.hpc_cmd_idx
            prefix = ">" if is_selected else " "
            style = term.bold if is_selected else term.normal
            max_label = left_width - 4
            disp_label = label[:max_label] if len(label) > max_label else label
            print(term.move_xy(1, y) + f"{prefix} {style}{disp_label}{term.normal}")

        # === Bottom-Left: Jobs ===
        jobs_start_y = left_split_y + 1
        pane_title = f" Jobs ({len(self.hpc_jobs)}) "
        if self.hpc_pane_focus == 1:
            print(term.move_xy(2, jobs_start_y) + term.reverse(pane_title))
        else:
            print(term.move_xy(2, jobs_start_y) + term.bold(pane_title))

        # Job list header
        header_y = jobs_start_y + 1
        print(
            term.move_xy(2, header_y)
            + term.bright_black(f"{'ID':<8} {'St':<4} {'Name'}")
        )

        # Job list
        job_start_y = header_y + 1
        max_job_rows = content_end_y - job_start_y

        if self.hpc_jobs:
            start_idx = max(0, self.hpc_job_idx - max_job_rows + 1) if self.hpc_pane_focus == 1 else 0
            visible_jobs = self.hpc_jobs[start_idx : start_idx + max_job_rows]

            for i, job in enumerate(visible_jobs):
                y = job_start_y + i
                real_idx = start_idx + i

                is_selected = real_idx == self.hpc_job_idx
                prefix = ">" if (self.hpc_pane_focus == 1 and is_selected) else " "
                style = term.bold if is_selected else term.normal

                status_color = (
                    term.green if job.status == "RUN"
                    else term.yellow if job.status == "PEND"
                    else term.white
                )

                max_name_len = left_width - 18
                name = job.name[:max_name_len] if len(job.name) > max_name_len else job.name

                line = f"{prefix}{job.id:<8} {status_color}{job.status:<4}{term.normal} {style}{name}{term.normal}"
                print(term.move_xy(1, y) + line)
        else:
            print(term.move_xy(3, job_start_y + 1) + term.bright_black("(No jobs)"))
            print(term.move_xy(3, job_start_y + 2) + term.bright_black("Auto-refresh: 5s"))

        # === Top-Right: Details ===
        pane_title = " Details "
        if self.hpc_pane_focus == 2:
            print(term.move_xy(sep_x + 2, content_start_y) + term.reverse(pane_title))
        else:
            print(term.move_xy(sep_x + 2, content_start_y) + term.bold(pane_title))

        details_width = term.width - sep_x - 3
        details_height = right_split_y - content_start_y - 2

        if self.hpc_details_lines:
            visible = self.hpc_details_lines[self.hpc_details_scroll : self.hpc_details_scroll + details_height]
            for i, line in enumerate(visible):
                y = content_start_y + 2 + i
                disp_line = line[:details_width] if len(line) > details_width else line
                print(term.move_xy(sep_x + 2, y) + disp_line)
        else:
            print(term.move_xy(sep_x + 4, content_start_y + 2) + term.bright_black("Select a job"))

        # === Bottom-Right: Output ===
        output_start_y = right_split_y + 1
        pane_title = " Output "
        if self.hpc_pane_focus == 3:
            print(term.move_xy(sep_x + 2, output_start_y) + term.reverse(pane_title))
        else:
            print(term.move_xy(sep_x + 2, output_start_y) + term.bold(pane_title))

        output_width = term.width - sep_x - 3
        output_height = content_end_y - output_start_y - 1

        if self.hpc_output_lines:
            visible = self.hpc_output_lines[self.hpc_output_scroll : self.hpc_output_scroll + output_height]
            for i, line in enumerate(visible):
                y = output_start_y + 2 + i
                if y >= content_end_y:
                    break
                disp_line = line[:output_width] if len(line) > output_width else line
                print(term.move_xy(sep_x + 2, y) + disp_line)

            # Scroll indicator
            if len(self.hpc_output_lines) > output_height:
                scroll_pct = int((self.hpc_output_scroll / max(1, len(self.hpc_output_lines) - output_height)) * 100)
                scroll_info = f"[{scroll_pct}%]"
                print(term.move_xy(term.width - len(scroll_info) - 1, content_end_y) + term.bright_black(scroll_info))
        else:
            print(term.move_xy(sep_x + 4, output_start_y + 2) + term.bright_black("(no output)"))

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
        if current_tab == "Actions":
            help_msg = " [h/l] Tabs | [j/k] Select | [Enter] Run | [G] Git | [q] Quit"
        elif current_tab == "HPC":
            help_msg = " [Tab] Pane | [j/k] Nav | [Enter] Run | [r] Refresh | [d] Kill | [G] Git | [q] Quit"

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

                if key.lower() == "q":
                    break

                # Global: Launch lazygit
                if key == "G":
                    subprocess.run(["lazygit"])
                    continue

                # Manual refresh
                if key.lower() == "r":
                    self.refresh_jobs()
                    continue

                # Tab navigation
                if key.name == "KEY_LEFT" or key == "h":
                    self.current_tab_idx = (self.current_tab_idx - 1) % len(self.tabs)
                elif key.name == "KEY_RIGHT" or key == "l":
                    self.current_tab_idx = (self.current_tab_idx + 1) % len(self.tabs)

                # --- Actions Tab Controls ---
                elif self.tabs[self.current_tab_idx] == "Actions":
                    if key.name == "KEY_UP" or key == "k":
                        self.action_cat_idx = max(0, self.action_cat_idx - 1)
                    elif key.name == "KEY_DOWN" or key == "j":
                        self.action_cat_idx = min(len(self.action_menus) - 1, self.action_cat_idx + 1)
                    elif key.name == "KEY_ENTER":
                        cat_name = list(self.action_menus.keys())[self.action_cat_idx]
                        items = self.action_menus[cat_name]

                        def _submenu(cat=cat_name, actions=items):
                            choices = [x[0] for x in actions] + ["Back"]
                            sel = questionary.select(
                                f"Select {cat} Action:",
                                choices=choices,
                                style=get_custom_style(),
                            ).ask()

                            if sel and sel != "Back":
                                for label, func in actions:
                                    if label == sel:
                                        func()
                                        input("\nPress Enter to return...")

                        return _submenu

                # --- HPC Tab Controls ---
                elif self.tabs[self.current_tab_idx] == "HPC":
                    if key.name == "KEY_TAB":
                        self.hpc_pane_focus = (self.hpc_pane_focus + 1) % 4
                    elif key.name == "KEY_UP" or key == "k":
                        if self.hpc_pane_focus == 0:
                            self.hpc_cmd_idx = max(0, self.hpc_cmd_idx - 1)
                        elif self.hpc_pane_focus == 1:
                            old_idx = self.hpc_job_idx
                            self.hpc_job_idx = max(0, self.hpc_job_idx - 1)
                            if old_idx != self.hpc_job_idx:
                                self.refresh_job_data()
                        elif self.hpc_pane_focus == 2:
                            self.hpc_details_scroll = max(0, self.hpc_details_scroll - 1)
                        else:  # Output pane
                            self.hpc_output_scroll = max(0, self.hpc_output_scroll - 1)
                    elif key.name == "KEY_DOWN" or key == "j":
                        if self.hpc_pane_focus == 0:
                            self.hpc_cmd_idx = min(len(self.hpc_commands) - 1, self.hpc_cmd_idx + 1)
                        elif self.hpc_pane_focus == 1:
                            old_idx = self.hpc_job_idx
                            self.hpc_job_idx = min(max(0, len(self.hpc_jobs) - 1), self.hpc_job_idx + 1)
                            if old_idx != self.hpc_job_idx:
                                self.refresh_job_data()
                        elif self.hpc_pane_focus == 2:
                            max_scroll = max(0, len(self.hpc_details_lines) - 5)
                            self.hpc_details_scroll = min(max_scroll, self.hpc_details_scroll + 1)
                        else:  # Output pane
                            max_scroll = max(0, len(self.hpc_output_lines) - 10)
                            self.hpc_output_scroll = min(max_scroll, self.hpc_output_scroll + 1)
                    elif key.name == "KEY_ENTER":
                        if self.hpc_pane_focus == 0:
                            # Execute command
                            _, factory = self.hpc_commands[self.hpc_cmd_idx]
                            return factory()
                        elif self.hpc_pane_focus == 1:
                            # Job selected - refresh data
                            self.refresh_job_data()
                    elif key == "d":
                        # Kill selected job
                        if self.hpc_jobs and self.hpc_job_idx < len(self.hpc_jobs):
                            from src.utils.tui.monitor import kill_job
                            job = self.hpc_jobs[self.hpc_job_idx]
                            kill_job(job.id)
                            self.refresh_jobs()


def run_tui():
    """Entry point to run the TUI application."""
    app = TuiApp()
    action = app.run()
    if action:
        action()
