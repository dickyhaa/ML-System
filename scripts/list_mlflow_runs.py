from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri='http://127.0.0.1:5000')
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experiment {exp.experiment_id}: {exp.name}")
    # search_runs returns a list of Run objects
    runs = client.search_runs([exp.experiment_id])
    for run in runs:
        info = run.info
        print(f"  run_id: {info.run_id}  status: {info.status}")
