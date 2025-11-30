"""Textual-based TUI application for project management."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Label,
    Log,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    DataTable,
    OptionList,
)
from textual.widgets.option_list import Option

from utils.config import load_project_config


class DescriptionPane(Static):
    """Shows description of selected item."""

    def update_description(self, title: str, lines: list[str]) -> None:
        content = f"[bold]{title}[/bold]\n\n" + "\n".join(lines)
        self.update(content)


class ProjectTUI(App):
    """Main TUI application."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("g", "open_git", "Git"),
        Binding("c", "clear_log", "Clear"),
    ]

    def __init__(self):
        super().__init__()
        self.actions = [
            {
                "id": "compute",
                "label": "Run Compute Scripts",
                "category": "Compute",
                "description": [
                    "Executes all compute scripts in the project.",
                    "Runs numerical simulations and data processing.",
                ],
            },
            {
                "id": "plots",
                "label": "Run Plot Scripts",
                "category": "Compute",
                "description": [
                    "Generates all plots from computed data.",
                    "Plots saved to configured output directory.",
                ],
            },
            {
                "id": "copy_plots",
                "label": "Copy Plots to Report",
                "category": "Compute",
                "description": [
                    "Copies generated plots to the LaTeX report",
                    "figures directory.",
                ],
            },
            {
                "id": "mlflow",
                "label": "Fetch MLflow Artifacts",
                "category": "Data",
                "description": [
                    "Downloads experiment artifacts from the",
                    "configured MLflow tracking server.",
                ],
            },
            {
                "id": "docs",
                "label": "Build Documentation",
                "category": "Docs",
                "description": [
                    "Builds project documentation using Sphinx.",
                    "Opens docs in browser when complete.",
                ],
            },
            {
                "id": "clean",
                "label": "Clean Generated Files",
                "category": "Maintenance",
                "description": [
                    "Removes all generated files including:",
                    "computed data, plots, and build artifacts.",
                ],
            },
        ]

        self.hpc_commands = [
            {
                "id": "generate",
                "label": "Generate Job Pack",
                "description": [
                    "Creates LSF job pack files from packs.yaml.",
                    "Select groups, generate with parameter sweeps.",
                ],
            },
            {
                "id": "submit",
                "label": "Submit Job Pack",
                "description": [
                    "Submit a generated .pack file to LSF.",
                    "Requires bsub command and LSF environment.",
                ],
            },
        ]

    def compose(self) -> ComposeResult:
        yield Footer()

        with TabbedContent(id="main-tabs"):
            # Actions Tab
            with TabPane("Actions", id="tab-actions"):
                with Horizontal(classes="main-layout"):
                    with Vertical(classes="left-pane"):
                        desc = DescriptionPane(id="action-desc", classes="desc-pane")
                        desc.border_title = "Description"
                        yield desc
                        
                        # Use OptionList for actions
                        options = [
                            Option(f"{a['label']} [{a['category']}]", id=a["id"])
                            for a in self.actions
                        ]
                        lst = OptionList(*options, id="action-list", classes="list-pane")
                        lst.border_title = "Actions"
                        lst.border_subtitle = "[↑/↓] Navigate • [Enter] Execute"
                        yield lst
                    
                    log = RichLog(id="output-log", classes="output-pane", markup=True)
                    log.border_title = "Output Log"
                    yield log

            # HPC Tab
            with TabPane("HPC", id="tab-hpc"):
                with Horizontal(classes="main-layout"):
                    with Vertical(classes="left-pane"):
                        desc = DescriptionPane(id="hpc-desc", classes="desc-pane")
                        desc.border_title = "Description"
                        yield desc
                        
                        # Use OptionList for HPC commands
                        hpc_options = [
                            Option(c["label"], id=c["id"])
                            for c in self.hpc_commands
                        ]
                        lst = OptionList(*hpc_options, id="command-list", classes="command-list")
                        lst.border_title = "HPC Commands"
                        yield lst
                        
                        with TabbedContent(id="job-tabs", classes="job-tabs"):
                            with TabPane("Active", id="jobs-active"):
                                table = DataTable(id="active-jobs-table", cursor_type="row")
                                table.add_columns("ID", "Name", "Status", "Queue", "Cores")
                                yield table
                            with TabPane("Pending", id="jobs-pending"):
                                table = DataTable(id="pending-jobs-table", cursor_type="row")
                                table.add_columns("ID", "Name", "Status", "Queue", "Cores")
                                yield table
                            with TabPane("Done", id="jobs-done"):
                                table = DataTable(id="done-jobs-table", cursor_type="row")
                                table.add_columns("ID", "Name", "Status", "Queue", "Cores")
                                yield table
                    
                    log = RichLog(id="hpc-log", classes="output-pane", markup=True)
                    log.border_title = "HPC Log"
                    yield log

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Set theme
        self.theme = "nord"
        
        # Set initial focus and selection
        try:
            action_list = self.query_one("#action-list", OptionList)
            action_list.highlighted = 0
            action_list.focus()
        except Exception:
            pass

        try:
            cmd_list = self.query_one("#command-list", OptionList)
            cmd_list.highlighted = 0
        except Exception:
            pass

        # Set initial descriptions
        self._update_action_description("compute") # Default first item
        self._update_hpc_description("generate")   # Default first item
        # Refresh jobs
        self.refresh_jobs()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Handle option list highlight changes."""
        if event.option_list.id == "action-list" and event.option_id:
            self._update_action_description(event.option_id)
        elif event.option_list.id == "command-list" and event.option_id:
            self._update_hpc_description(event.option_id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option list selection (Enter key)."""
        if event.option_list.id == "action-list" and event.option_id:
            self.execute_project_action(event.option_id)
        elif event.option_list.id == "command-list" and event.option_id:
            self.run_hpc_command(event.option_id)

    def _update_action_description(self, action_id: str) -> None:
        """Update action description pane."""
        action = next((a for a in self.actions if a["id"] == action_id), None)
        if action:
            desc_pane = self.query_one("#action-desc", DescriptionPane)
            desc_pane.update_description(action["label"], action["description"])

    def _update_hpc_description(self, command_id: str) -> None:
        """Update HPC command description pane."""
        cmd = next((c for c in self.hpc_commands if c["id"] == command_id), None)
        if cmd:
            desc_pane = self.query_one("#hpc-desc", DescriptionPane)
            desc_pane.update_description(cmd["label"], cmd["description"])

    def write_log(self, message: str, pane: str = "output") -> None:
        """Write to the appropriate log pane."""
        if pane == "output":
            log_widget = self.query_one("#output-log", RichLog)
        else:
            log_widget = self.query_one("#hpc-log", RichLog)
        log_widget.write(message)

    def write_header(self, title: str, pane: str = "output") -> None:
        """Write a header to the log."""
        self.write_log(f"\n[bold cyan]─── {title} ───[/bold cyan]", pane)

    # --- Action Handlers ---

    def execute_project_action(self, action_id: str) -> None:
        """Run the specified action."""
        from utils import runners, mlflow
        from utils.config import clean_all
        from utils.tui.actions import docs
        import subprocess

        self.write_header(action_id.replace("_", " ").title())
        self.write_log("Running...")

        try:
            if action_id == "compute":
                runners.run_compute_scripts()
                self.write_log("[green]✓ Compute scripts completed[/green]")

            elif action_id == "plots":
                runners.run_plot_scripts()
                self.write_log("[green]✓ Plot scripts completed[/green]")

            elif action_id == "copy_plots":
                runners.copy_to_report()
                self.write_log("[green]✓ Plots copied to report[/green]")

            elif action_id == "mlflow":
                config = load_project_config()
                mlflow_conf = config.get("mlflow", {})
                mlflow.setup_mlflow_tracking()
                output_dir = Path.cwd() / mlflow_conf.get("download_dir", "data/downloaded")
                if mlflow_conf.get("databricks_dir"):
                    mlflow.fetch_project_artifacts(output_dir)
                    self.write_log("[green]✓ MLflow artifacts fetched[/green]")
                else:
                    self.write_log("[yellow]No databricks_dir configured in project_config.yaml[/yellow]")

            elif action_id == "docs":
                success = docs.build_docs()
                if success:
                    docs_index = Path.cwd() / "docs" / "build" / "html" / "index.html"
                    if docs_index.exists():
                        subprocess.run(["xdg-open", str(docs_index)])
                        self.write_log("[green]✓ Docs built and opened[/green]")
                else:
                    self.write_log("[red]✗ Docs build failed[/red]")

            elif action_id == "clean":
                clean_all()
                self.write_log("[green]✓ Cleaned generated files[/green]")

        except Exception as e:
            self.write_log(f"[red]Error: {e}[/red]")

    # --- HPC Handlers ---

    def run_hpc_command(self, command_id: str) -> None:
        """Run the specified HPC command."""
        if command_id == "generate":
            self.action_generate_pack()
        elif command_id == "submit":
            self.action_submit_pack()

    def action_generate_pack(self) -> None:
        """Generate job pack files."""
        from utils.hpc import get_available_groups
        from utils.hpc.submit import generate_pack

        config_path, job_packs_dir = self._get_hpc_paths()

        groups = get_available_groups(config_path)
        if not groups:
            self.write_log("[yellow]No groups found in packs.yaml[/yellow]", "hpc")
            return

        # For now, generate all groups
        # TODO: Add group selection screen
        self.write_header("Generate Job Pack", "hpc")
        self.write_log(f"Groups: {', '.join(groups)}", "hpc")

        files, log_lines = generate_pack(config_path, job_packs_dir, groups)

        for line in log_lines:
            self.write_log(line, "hpc")

        if files:
            self.write_log(f"[green]✓ Generated {len(files)} pack file(s)[/green]", "hpc")
        else:
            self.write_log("[red]✗ No files generated[/red]", "hpc")

    def action_submit_pack(self) -> None:
        """Submit a pack file to LSF."""
        from utils.hpc import get_pack_files, submit_pack

        _, job_packs_dir = self._get_hpc_paths()

        pack_files = get_pack_files(job_packs_dir)
        if not pack_files:
            self.write_log(f"[yellow]No .pack files in {job_packs_dir}[/yellow]", "hpc")
            return

        # For now, submit the first/newest pack
        # TODO: Add pack selection screen
        pack_file = pack_files[0]

        self.write_header("Submit Job Pack", "hpc")
        self.write_log(f"Submitting: {pack_file.name}", "hpc")

        success, log_lines = submit_pack(pack_file)

        for line in log_lines:
            self.write_log(line, "hpc")

        if success:
            self.write_log("[green]✓ Submission successful[/green]", "hpc")
            self.refresh_jobs()
        else:
            self.write_log("[red]✗ Submission failed[/red]", "hpc")

    def _get_hpc_paths(self) -> tuple[Path, Path]:
        """Get HPC config paths."""
        repo_root = Path.cwd()
        config = load_project_config()
        hpc_config = config.get("hpc", {})
        job_packs_dir_str = hpc_config.get("job_packs", "Experiments/05-scaling/job-packs")
        job_packs_dir = repo_root / job_packs_dir_str
        config_path = job_packs_dir / "packs.yaml"
        return config_path, job_packs_dir

    def refresh_jobs(self) -> None:
        """Refresh job lists from LSF."""
        from utils.tui.monitor import get_jobs

        jobs = get_jobs()

        active = [j for j in jobs if j.status == "RUN"]
        pending = [j for j in jobs if j.status == "PEND"]
        done = [j for j in jobs if j.status not in ("RUN", "PEND")]

        # Update job lists
        self._update_job_list("#active-jobs-table", active)
        self._update_job_list("#pending-jobs-table", pending)
        self._update_job_list("#done-jobs-table", done)

    def _update_job_list(self, table_id: str, jobs: list) -> None:
        """Update a job list widget."""
        try:
            table = self.query_one(table_id, DataTable)
            table.clear()
            for job in jobs:
                # Handle potentially missing fields safely
                queue = getattr(job, "queue", "?")
                cores = getattr(job, "cores", "?")
                table.add_row(job.id, job.name, job.status, queue, cores, key=job.id)
        except Exception:
            pass  # Table might not exist yet

    # --- Actions ---

    def action_refresh(self) -> None:
        """Refresh jobs."""
        self.refresh_jobs()
        self.write_log("[dim]Refreshed[/dim]")

    def action_open_git(self) -> None:
        """Open lazygit."""
        import subprocess
        self.suspend()
        subprocess.run(["lazygit"])
        self.resume()

    def action_clear_log(self) -> None:
        """Clear current log."""
        try:
            self.query_one("#output-log", RichLog).clear()
        except Exception:
            pass
        try:
            self.query_one("#hpc-log", RichLog).clear()
        except Exception:
            pass


def run_tui() -> None:
    """Entry point to run the TUI application."""
    app = ProjectTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
