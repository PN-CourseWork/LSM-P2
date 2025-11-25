
def mlflow_start_log(self, experiment_name, N, max_iter, tolerance):
    mlflow.login(backend="databricks", interactive=False)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

    # Update config with runtime values and log immediately
    self.config.N = N
    self.config.max_iter = max_iter
    self.config.tolerance = tolerance
    mlflow.log_params(asdict(self.config))

def mlflow_end_log(self):

    # Log global results (excluding lists which can't be logged as metrics)
    global_dict = asdict(self.global_results)
    residual_history = global_dict.pop('residual_history', [])

    mlflow.log_metrics(global_dict)

    # Log residual history as step-by-step metrics for convergence graph
    for step, residual in enumerate(residual_history):
        mlflow.log_metric("residual", residual, step=step)

    # Log per-rank results as a table
    per_rank_dicts = [asdict(pr) for pr in self.all_per_rank_results]
    mlflow.log_table(pd.DataFrame(per_rank_dicts), "per_rank_results.json")
    mlflow.end_run()

