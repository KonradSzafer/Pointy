import numpy as np


class MetricTracker:
    def __init__(self):
        self.metrics_fn = {}
        self.reset()

    def reset(self) -> None:
        self.batch_metrics = []
        self.aggregated_metrics = {}

    def register_metric(self, name: str, metric: callable) -> None:
        self.metrics_fn[name] = metric

    def update(self, *args) -> dict[str, float]:
        batch_results = {}
        for name, metric in self.metrics_fn.items():
            batch_results[name] = metric(*args).item()
        self.batch_metrics.append(batch_results)
        return batch_results

    def aggregate(self) -> dict[str, float]:
        for name in self.metrics_fn.keys():
            self.aggregated_metrics[name] = float(
                np.mean([batch[name] for batch in self.batch_metrics])
            )
        return self.aggregated_metrics
