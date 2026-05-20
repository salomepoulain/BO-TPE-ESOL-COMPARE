"""Persistence layer for optimization runs and trial snapshots.

This module stores trial lifecycle events in SQLite and exports JSON/CSV views
for downstream analysis. The schema is intentionally method-agnostic so random,
BoTorch, and Optuna runs can be inspected with one reader path.

The storage API is append/update oriented to support resumable experiments and
to make partial progress visible during long-running searches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


TRIAL_COLUMNS = (
    "method",
    "trial",
    "status",
    "validation_loss",
    "num_layers",
    "hidden_dim",
    "layer_type",
    "activation",
    "pooling",
    "optimizer_type",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "elapsed_seconds",
    "error",
)


class OptimizationStorage:
    """SQLite-backed optimization run storage with CSV/JSON snapshots."""

    def __init__(
        self,
        *,
        method: str,
        output_dir: str | Path = "output",
        run_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.run_folder = Path(output_dir) / (run_name or default_run_name(method))
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_folder / "trials.sqlite"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_tables()
        self.update_metadata({"method": method, **dict(metadata or {})})

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Merge metadata into the SQLite store and metadata.json snapshot."""
        safe_metadata = _json_safe(dict(metadata))
        with self.connection:
            for key, value in safe_metadata.items():
                self.connection.execute(
                    """
                    insert into metadata(key, value_json)
                    values (?, ?)
                    on conflict(key) do update set value_json = excluded.value_json
                    """,
                    (key, json.dumps(value)),
                )
        (self.run_folder / "metadata.json").write_text(
            json.dumps(self.metadata(), indent=2),
            encoding="utf-8",
        )

    def metadata(self) -> dict[str, Any]:
        """Return stored run metadata.

        Returns
        -------
        dict[str, Any]
            Metadata key/value mapping.
        """
        rows = self.connection.execute("select key, value_json from metadata").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def save_pending(
        self,
        *,
        trial: int,
        parameters: Mapping[str, Any],
        x: Sequence[float] | None = None,
    ) -> None:
        """Save a candidate before objective evaluation starts."""
        self._save_trial(
            trial=trial,
            status="pending",
            parameters=parameters,
            x=x,
        )
        self.export()

    def save_completed(
        self,
        *,
        trial: int,
        parameters: Mapping[str, Any],
        validation_loss: float,
        elapsed_seconds: float,
        x: Sequence[float] | None = None,
    ) -> None:
        """Save a successfully evaluated trial."""
        self._save_trial(
            trial=trial,
            status="completed",
            parameters=parameters,
            validation_loss=validation_loss,
            elapsed_seconds=elapsed_seconds,
            x=x,
        )
        self.export()

    def save_failed(
        self,
        *,
        trial: int,
        parameters: Mapping[str, Any],
        error: str,
        elapsed_seconds: float,
        x: Sequence[float] | None = None,
    ) -> None:
        """Save a failed trial, keeping enough information for diagnosis."""
        self._save_trial(
            trial=trial,
            status="failed",
            parameters=parameters,
            elapsed_seconds=elapsed_seconds,
            error=error,
            x=x,
        )
        self.export()

    def completed_trials(self) -> list[dict[str, Any]]:
        """Return completed trials with finite validation loss.

        Returns
        -------
        list[dict[str, Any]]
            Completed trial rows ordered by trial index.
        """
        return [
            row
            for row in self.trials()
            if row["status"] == "completed" and row.get("validation_loss") is not None
        ]

    def trials(self) -> list[dict[str, Any]]:
        """Return all saved trials.

        Returns
        -------
        list[dict[str, Any]]
            Trial rows ordered by trial index.
        """
        rows = self.connection.execute("select * from trials order by trial").fetchall()
        return [self._row_to_trial(row) for row in rows]

    def export(self) -> None:
        """Write JSON and CSV trial snapshots from SQLite."""
        rows = self.trials()
        (self.run_folder / "trials.json").write_text(
            json.dumps(_json_safe(rows), indent=2),
            encoding="utf-8",
        )
        _write_csv(self.run_folder / "trials.csv", rows)

    def _create_tables(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                create table if not exists metadata (
                    key text primary key,
                    value_json text not null
                )
                """
            )
            self.connection.execute(
                """
                create table if not exists trials (
                    trial integer primary key,
                    method text not null,
                    status text not null,
                    parameters_json text not null,
                    x_json text,
                    validation_loss real,
                    elapsed_seconds real,
                    error text,
                    updated_at text not null
                )
                """
            )

    def _save_trial(
        self,
        *,
        trial: int,
        status: str,
        parameters: Mapping[str, Any],
        validation_loss: float | None = None,
        elapsed_seconds: float | None = None,
        error: str | None = None,
        x: Sequence[float] | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                insert into trials(
                    trial, method, status, parameters_json, x_json,
                    validation_loss, elapsed_seconds, error, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(trial) do update set
                    method = excluded.method,
                    status = excluded.status,
                    parameters_json = excluded.parameters_json,
                    x_json = excluded.x_json,
                    validation_loss = excluded.validation_loss,
                    elapsed_seconds = excluded.elapsed_seconds,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    trial,
                    self.method,
                    status,
                    json.dumps(_json_safe(dict(parameters))),
                    json.dumps(list(x)) if x is not None else None,
                    validation_loss,
                    elapsed_seconds,
                    error,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def _row_to_trial(self, row: sqlite3.Row) -> dict[str, Any]:
        parameters = json.loads(row["parameters_json"])
        trial = {
            "method": row["method"],
            "trial": row["trial"],
            "status": row["status"],
            "validation_loss": row["validation_loss"],
            **parameters,
            "elapsed_seconds": row["elapsed_seconds"],
            "error": row["error"],
        }
        if row["x_json"] is not None:
            trial["x"] = json.loads(row["x_json"])
        return trial


def default_run_name(method: str) -> str:
    """Build a stable default run name.

    Parameters
    ----------
    method : str
        Optimization method name.

    Returns
    -------
    str
        Derived run folder name.
    """
    safe_method = method.lower().replace(" ", "_")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id and slurm_task_id:
        return f"{safe_method}-slurm-{slurm_job_id}-{slurm_task_id}"
    if slurm_job_id:
        return f"{safe_method}-slurm-{slurm_job_id}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{safe_method}-{timestamp}"


def save_optimisation(
    *,
    method: str,
    trials: Sequence[Mapping[str, Any]],
    output_dir: str | Path = "output",
    metadata: Mapping[str, Any] | None = None,
    run_name: str | None = None,
) -> Path:
    """Save optimization rows into an output run folder.

    Returns
    -------
    Path
        Path to the run folder.
    """
    if not trials:
        raise ValueError("trials cannot be empty")
    storage = OptimizationStorage(
        method=method,
        output_dir=output_dir,
        run_name=run_name,
        metadata=metadata,
    )
    for trial in trials:
        validation_loss = trial.get("validation_loss")
        status = trial.get("status", "completed")
        if status == "completed" and validation_loss is not None:
            storage.save_completed(
                trial=int(trial["trial"]),
                parameters=_trial_parameters(trial),
                validation_loss=float(validation_loss),
                elapsed_seconds=float(trial.get("elapsed_seconds") or 0.0),
                x=trial.get("x"),
            )
        else:
            storage.save_pending(
                trial=int(trial["trial"]),
                parameters=_trial_parameters(trial),
                x=trial.get("x"),
            )
    return storage.run_folder


def _trial_parameters(trial: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in trial.items()
        if key not in {"method", "trial", "status", "validation_loss", "elapsed_seconds", "error", "x"}
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(
        dict.fromkeys(
            [
                *TRIAL_COLUMNS,
                *(key for row in rows for key in row if key not in TRIAL_COLUMNS),
            ]
        )
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value
