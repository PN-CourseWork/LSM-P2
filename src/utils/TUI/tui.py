"""Main TUI application using blessed."""

from blessed import Terminal
from src.utils import manage, mlflow_io
from src.utils.hpc import _interactive_generate, _interactive_submit
from src.utils.TUI.monitor import run_monitor, get_jobs
from src.utils.manage import load_project_config
from src.utils.TUI.actions import run, data, docs, clean
from pathlib import Path
import sys
import os
import subprocess
import questionary
from src.utils.TUI.styles import get_custom_style

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class TuiApp:
    def __init__(self):
        self.term = Terminal()
        # Redefined Tabs Structure
        self.tabs = ["HPC", "Actions", "Monitor"]
        self.current_tab_idx = 0
        
        self.hpc_menus = {
            "Commands": [
                ("Generate Job Pack", self.action_generate_pack),
                ("Submit Job Pack", self.action_submit_pack),
            ]
        }
        
        # HPC State
        self.hpc_pane_focus = 0 # 0: Commands, 1: Jobs
        self.hpc_cmd_idx = 0
        self.hpc_job_idx = 0
        self.hpc_jobs = [] # Cache for jobs list
        self.hpc_preview_content = ["Select an item to see details."]

        # Action Menus Mapping
        self.action_menus = {
            "Compute & Plotting": [
                ("Run Compute Scripts", lambda: manage.run_compute_scripts()),
                ("Run Plot Scripts", lambda: manage.run_plot_scripts()),
                ("Copy Plots to Report", lambda: manage.copy_plots()),
            ],
            "Data & MLflow": [
                ("Fetch Artifacts", self.action_fetch_mlflow()),
            ],
            "Documentation": [
                ("Build Docs", lambda: manage.build_docs()),
            ],
            "Maintenance": [
                ("Clean Generated Files", self.action_clean_all()),
            ]
        }
        self.action_cat_idx = 0
        self.action_item_idx = 0 # Not really used as we use questionary for sub-selection

    def get_project_paths(self):
        repo_root = Path.cwd()
        project_config = load_project_config()
        hpc_config = project_config.get("hpc", {})
        job_packs_dir_str = hpc_config.get("job_packs", "Experiments/05-scaling/job-packs")
        job_packs_dir = repo_root / job_packs_dir_str
        config_path = job_packs_dir / "packs.yaml"
        return config_path, job_packs_dir, repo_root, project_config

    # --- Actions Factories ---
    
    def action_generate_pack(self):
        config_path, job_packs_dir, _, _ = self.get_project_paths()
        return lambda: _interactive_generate(config_path, job_packs_dir)

    def action_submit_pack(self):
        _, job_packs_dir, _, _ = self.get_project_paths()
        return lambda: _interactive_submit(job_packs_dir)

    def action_fetch_mlflow(self):
        def _fetch():
            _, _, repo_root, config = self.get_project_paths()
            mlflow_conf = config.get("mlflow", {})
            mlflow_io.setup_mlflow_auth(mlflow_conf.get("tracking_uri"))
            output_dir = repo_root / mlflow_conf.get("download_dir", "data/downloaded")
            experiments = mlflow_conf.get("experiments", [])
            if not experiments:
                print("  (No experiments configured)")
            else:
                mlflow_io.fetch_project_artifacts(experiments, output_dir)
        return _fetch

    def action_clean_all(self):
        def _clean():
            if questionary.confirm("Are you sure?", style=get_custom_style()).ask():
                manage.clean_all()
        return _clean

    def refresh_jobs(self):
        """Fetch jobs for the HPC bottom pane."""
        self.hpc_jobs = get_jobs()

    def draw_hpc_dashboard(self):
        term = self.term
        # Layout:
        # Header takes top 4 lines (0-3)
        # Status line takes bottom 1 line (height-1)
        # Content area: y=4 to y=height-2
        
        content_start_y = 4
        content_end_y = term.height - 2
        available_height = content_end_y - content_start_y + 1
        
        mid_x = int(term.width * 0.4)
        # Split vertically roughly in half
        mid_y = content_start_y + int(available_height * 0.5)
        
        # Draw Vertical Separator
        for y in range(content_start_y, content_end_y + 1):
            print(term.move_xy(mid_x, y) + term.cyan("│"))
            
        # Draw Horizontal Separator (Left only)
        print(term.move_xy(0, mid_y) + term.cyan("─" * mid_x))
        print(term.move_xy(mid_x, mid_y) + term.cyan("┤"))

        # --- Top Left: Commands ---
        print(term.move_xy(2, content_start_y) + term.bold_underline("Commands"))
        cmds = self.hpc_menus["Commands"]
        for i, (label, _) in enumerate(cmds):
            y = content_start_y + 2 + i
            if y < mid_y:
                prefix = " > " if (self.hpc_pane_focus == 0 and i == self.hpc_cmd_idx) else "   "
                style = term.bold if (self.hpc_pane_focus == 0 and i == self.hpc_cmd_idx) else term.normal
                print(term.move_xy(0, y) + f"{prefix}{style}{label}{term.normal}")

        # --- Bottom Left: Running Jobs ---
        print(term.move_xy(2, mid_y + 1) + term.bold_underline(f"Running Jobs ({len(self.hpc_jobs)})"))
        
        # Header
        print(term.move_xy(1, mid_y + 2) + term.bright_black(f"{ 'ID':<8} {'Status':<6} {'Name'}"))
        
        job_list_start_y = mid_y + 3
        max_job_rows = content_end_y - job_list_start_y + 1
        
        start_idx = max(0, self.hpc_job_idx - max_job_rows + 1) if self.hpc_pane_focus == 1 else 0
        visible_jobs = self.hpc_jobs[start_idx : start_idx + max_job_rows]
        
        for i, job in enumerate(visible_jobs):
            y = job_list_start_y + i
            real_idx = start_idx + i
            
            if y <= content_end_y:
                is_selected = (self.hpc_pane_focus == 1 and real_idx == self.hpc_job_idx)
                prefix = " > " if is_selected else "   "
                style = term.bold if is_selected else term.normal
                status_color = term.green if job.status == "RUN" else term.yellow if job.status == "PEND" else term.white
                
                # Truncate name
                max_name_len = mid_x - 18
                name = job.name[:max_name_len]
                
                line = f"{prefix}{job.id:<8} {status_color}{job.status:<6}{term.normal} {style}{name}{term.normal}"
                print(term.move_xy(0, y) + line)

        if not self.hpc_jobs:
             print(term.move_xy(4, mid_y + 4) + term.bright_black("(No jobs)"))

        # --- Right Pane: Preview ---
        print(term.move_xy(mid_x + 2, content_start_y) + term.bold_underline("Preview / Info"))
        
        # Determine content
        content = []
        if self.hpc_pane_focus == 0:
            # Command Info
            label = cmds[self.hpc_cmd_idx][0]
            if "Generate" in label:
                content = ["Generates LSF job script files based on", "packs.yaml configuration.", "", "Interactive wizard will ask for groups."]
            else:
                content = ["Submits generated .pack files to the", "LSF scheduler using 'bsub'.", "", "You can review the file before submitting."]
        else:
            # Job Info
            if self.hpc_jobs and self.hpc_job_idx < len(self.hpc_jobs):
                j = self.hpc_jobs[self.hpc_job_idx]
                content = [
                    f"Job ID:    {j.id}",
                    f"Name:      {j.name}",
                    f"Status:    {j.status}",
                    f"Queue:     {j.queue}",
                    f"Cores:     {j.cores}",
                    f"Started:   {j.start_time}",
                    f"Elapsed:   {j.elapsed}",
                ]
            else:
                content = ["No job selected."]

        for i, line in enumerate(content):
            print(term.move_xy(mid_x + 2, content_start_y + 2 + i) + line)


    def draw_actions_dashboard(self):
        term = self.term
        categories = list(self.action_menus.keys())
        
        # Center layout
        start_y = 5
        
        for i, cat in enumerate(categories):
            prefix = " > " if i == self.action_cat_idx else "   "
            style = term.bold if i == self.action_cat_idx else term.normal
            print(term.move_xy(4, start_y + i*2) + f"{prefix}{style}{cat}{term.normal}")
            
        # Right side description
        mid_x = int(term.width * 0.4)
        print(term.move_xy(mid_x, 3) + term.cyan("│"))
        for y in range(3, term.height - 1): # Extend closer to bottom
            print(term.move_xy(mid_x, y) + term.cyan("│"))
            
        cat_name = categories[self.action_cat_idx]
        items = self.action_menus[cat_name]
        
        print(term.move_xy(mid_x + 2, 5) + term.bold_underline(cat_name + " Actions"))
        for i, (label, _) in enumerate(items):
            print(term.move_xy(mid_x + 4, 7 + i) + f"• {label}")
            
        print(term.move_xy(mid_x + 2, 7 + len(items) + 2) + term.bright_black("Press Enter to select action"))


    def run(self):
        """Main TUI Loop."""
        # Initial data fetch
        self.refresh_jobs()
        
        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            while True:
                self.draw()
                key = self.term.inkey(timeout=5) # Refresh polling
                
                # Auto-refresh jobs every 5s
                if not key:
                    if self.tabs[self.current_tab_idx] == "HPC":
                        self.refresh_jobs()
                    continue

                if key.lower() == 'q':
                    break

                # --- Global Commands ---
                if key == 'G':
                    # Launch lazygit fullscreen
                    subprocess.run(['lazygit'])
                    continue

                # --- Global Navigation ---
                if key.name == 'KEY_LEFT' or key == 'h':
                    # Tab Switching
                    self.current_tab_idx = (self.current_tab_idx - 1) % len(self.tabs)
                elif key.name == 'KEY_RIGHT' or key == 'l':
                    self.current_tab_idx = (self.current_tab_idx + 1) % len(self.tabs)
                
                # --- Tab Specific Controls ---
                elif self.tabs[self.current_tab_idx] == "HPC":
                    if key.name == 'KEY_TAB':
                        # Switch Pane
                        self.hpc_pane_focus = 1 - self.hpc_pane_focus
                    elif key.name == 'KEY_UP' or key == 'k':
                        if self.hpc_pane_focus == 0:
                            self.hpc_cmd_idx = max(0, self.hpc_cmd_idx - 1)
                        else:
                            self.hpc_job_idx = max(0, self.hpc_job_idx - 1)
                    elif key.name == 'KEY_DOWN' or key == 'j':
                        if self.hpc_pane_focus == 0:
                            max_cmd = len(self.hpc_menus["Commands"]) - 1
                            self.hpc_cmd_idx = min(max_cmd, self.hpc_cmd_idx + 1)
                        else:
                            max_job = max(0, len(self.hpc_jobs) - 1)
                            self.hpc_job_idx = min(max_job, self.hpc_job_idx + 1)
                    elif key.name == 'KEY_ENTER':
                        if self.hpc_pane_focus == 0:
                            label, factory = self.hpc_menus["Commands"][self.hpc_cmd_idx]
                            return factory()
                        
                elif self.tabs[self.current_tab_idx] == "Actions":
                    if key.name == 'KEY_UP' or key == 'k':
                        self.action_cat_idx = max(0, self.action_cat_idx - 1)
                    elif key.name == 'KEY_DOWN' or key == 'j':
                        self.action_cat_idx = min(len(self.action_menus) - 1, self.action_cat_idx + 1)
                    elif key.name == 'KEY_ENTER':
                        # Trigger submenu for category
                        cat_name = list(self.action_menus.keys())[self.action_cat_idx]
                        items = self.action_menus[cat_name]
                        
                        # Return a wrapper that runs questionary selection
                        def _submenu():
                            from src.utils.TUI.styles import get_custom_style
                            choices = [x[0] for x in items] + ["Back"]
                            sel = questionary.select(
                                f"Select {cat_name} Action:",
                                choices=choices,
                                style=get_custom_style()
                            ).ask()
                            
                            if sel and sel != "Back":
                                # Find action
                                for label, func in items:
                                    if label == sel:
                                        func()
                                        input("\nPress Enter to return...")
                        
                        return _submenu

                elif self.tabs[self.current_tab_idx] == "Monitor":
                    if key.name == 'KEY_ENTER':
                        return lambda: run_monitor(self.term)


    def draw(self):
        print(self.term.home + self.term.clear)
        
        # Header
        header = f" {self.term.bold}LSM Project Manager{self.term.normal} "
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
        
        if current_tab == "HPC":
            self.draw_hpc_dashboard()
        elif current_tab == "Actions":
            self.draw_actions_dashboard()
        elif current_tab == "Monitor":
            print("\n" * 5)
            msg = "Press [Enter] to launch full-screen Live Monitor"
            print(msg.center(self.term.width))

        # Status Line (Bottom)
        if current_tab == "HPC":
            help_msg = " [Tab] Pane | [j/k] Nav | [Enter] Select | [G] Git | [q] Quit"
        elif current_tab == "Actions":
            help_msg = " [h/l] Tabs | [j/k] Select | [Enter] Run | [G] Git | [q] Quit"
        elif current_tab == "Monitor":
            help_msg = " [Enter] Launch | [h/l] Tabs | [G] Git | [q] Quit"
        
        status_bar = f"{help_msg:<{self.term.width}}"
        print(self.term.move_xy(0, self.term.height - 1) + self.term.black_on_white(status_bar), end="", flush=True)


def run_tui():
    """Entry point to run the TUI application."""
    app = TuiApp()
    action = app.run()
    if action:
        action()