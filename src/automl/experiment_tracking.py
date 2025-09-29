"""
Experiment Tracking & Management System

This module provides comprehensive experiment tracking capabilities including:
- Experiment logging and metadata storage
- Parameter and metric tracking
- Artifact management (models, plots, data)
- Experiment history and comparison
- Database persistence with SQLite
- Search and filtering capabilities
- Performance analytics and trends
"""

import sqlite3
import json
import pickle
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import logging
import shutil
import os

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    experiment_name: str
    description: str = ""
    tags: List[str] = None
    dataset_name: str = ""
    dataset_hash: str = ""
    target_column: str = ""
    task_type: str = ""  # classification, regression
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass 
class ExperimentMetrics:
    """Container for experiment metrics"""
    primary_metric: float
    primary_metric_name: str
    cv_mean: float = None
    cv_std: float = None
    train_time: float = None
    predict_time: float = None
    additional_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}

@dataclass
class ExperimentArtifacts:
    """Container for experiment artifacts"""
    model_path: str = None
    plots: Dict[str, str] = None  # plot_name -> file_path
    data_files: Dict[str, str] = None  # data_name -> file_path
    logs: List[str] = None
    
    def __post_init__(self):
        if self.plots is None:
            self.plots = {}
        if self.data_files is None:
            self.data_files = {}
        if self.logs is None:
            self.logs = []

@dataclass
class ExperimentRecord:
    """Complete experiment record"""
    experiment_id: str
    config: ExperimentConfig
    parameters: Dict[str, Any]
    metrics: ExperimentMetrics
    artifacts: ExperimentArtifacts
    timestamp: datetime
    duration: float = 0.0
    status: str = "completed"  # running, completed, failed
    notes: str = ""

class ExperimentDatabase:
    """
    SQLite database manager for experiment persistence.
    
    Features:
    - Automatic schema creation and migration
    - Efficient storage and retrieval
    - Full-text search capabilities
    - Data integrity and validation
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize experiment database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,  -- JSON array
                    dataset_name TEXT,
                    dataset_hash TEXT,
                    target_column TEXT,
                    task_type TEXT,
                    timestamp DATETIME,
                    duration REAL,
                    status TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS parameters (
                    experiment_id TEXT,
                    param_name TEXT,
                    param_value TEXT,  -- JSON serialized
                    param_type TEXT,   -- str, int, float, bool, dict, list
                    PRIMARY KEY (experiment_id, param_name),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    is_primary BOOLEAN DEFAULT FALSE,
                    PRIMARY KEY (experiment_id, metric_name),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                
                CREATE TABLE IF NOT EXISTS artifacts (
                    experiment_id TEXT,
                    artifact_name TEXT,
                    artifact_path TEXT,
                    artifact_type TEXT,  -- model, plot, data, log
                    file_size INTEGER,
                    PRIMARY KEY (experiment_id, artifact_name),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp);
                CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(experiment_name);
                CREATE INDEX IF NOT EXISTS idx_experiments_task_type ON experiments(task_type);
                CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_name);
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_parameters_name ON parameters(param_name);
            """)
            
        self.logger.info(f"Experiment database initialized at {self.db_path}")
    
    def save_experiment(self, experiment: ExperimentRecord) -> bool:
        """
        Save experiment record to database.
        
        Args:
            experiment: Complete experiment record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insert experiment
                conn.execute("""
                    INSERT OR REPLACE INTO experiments 
                    (experiment_id, experiment_name, description, tags, dataset_name, 
                     dataset_hash, target_column, task_type, timestamp, duration, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment.experiment_id,
                    experiment.config.experiment_name,
                    experiment.config.description,
                    json.dumps(experiment.config.tags),
                    experiment.config.dataset_name,
                    experiment.config.dataset_hash,
                    experiment.config.target_column,
                    experiment.config.task_type,
                    experiment.timestamp.isoformat(),
                    experiment.duration,
                    experiment.status,
                    experiment.notes
                ))
                
                # Insert parameters
                for param_name, param_value in experiment.parameters.items():
                    param_type = type(param_value).__name__
                    conn.execute("""
                        INSERT OR REPLACE INTO parameters 
                        (experiment_id, param_name, param_value, param_type)
                        VALUES (?, ?, ?, ?)
                    """, (
                        experiment.experiment_id,
                        param_name,
                        json.dumps(param_value),
                        param_type
                    ))
                
                # Insert metrics
                metrics_dict = asdict(experiment.metrics)
                for metric_name, metric_value in metrics_dict.items():
                    if metric_value is not None and isinstance(metric_value, (int, float)):
                        is_primary = (metric_name == 'primary_metric')
                        conn.execute("""
                            INSERT OR REPLACE INTO metrics 
                            (experiment_id, metric_name, metric_value, is_primary)
                            VALUES (?, ?, ?, ?)
                        """, (
                            experiment.experiment_id,
                            metric_name,
                            float(metric_value),
                            is_primary
                        ))
                
                # Insert additional metrics
                if experiment.metrics.additional_metrics:
                    for metric_name, metric_value in experiment.metrics.additional_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            conn.execute("""
                                INSERT OR REPLACE INTO metrics 
                                (experiment_id, metric_name, metric_value, is_primary)
                                VALUES (?, ?, ?, ?)
                            """, (
                                experiment.experiment_id,
                                metric_name,
                                float(metric_value),
                                False
                            ))
                
                # Insert artifacts
                artifacts_dict = asdict(experiment.artifacts)
                for artifact_type, artifacts in artifacts_dict.items():
                    if artifacts:
                        if isinstance(artifacts, str):  # single file
                            if Path(artifacts).exists():
                                file_size = Path(artifacts).stat().st_size
                                conn.execute("""
                                    INSERT OR REPLACE INTO artifacts 
                                    (experiment_id, artifact_name, artifact_path, artifact_type, file_size)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    experiment.experiment_id,
                                    artifact_type,
                                    artifacts,
                                    artifact_type,
                                    file_size
                                ))
                        elif isinstance(artifacts, dict):  # multiple files
                            for artifact_name, artifact_path in artifacts.items():
                                if Path(artifact_path).exists():
                                    file_size = Path(artifact_path).stat().st_size
                                    conn.execute("""
                                        INSERT OR REPLACE INTO artifacts 
                                        (experiment_id, artifact_name, artifact_path, artifact_type, file_size)
                                        VALUES (?, ?, ?, ?, ?)
                                    """, (
                                        experiment.experiment_id,
                                        artifact_name,
                                        artifact_path,
                                        artifact_type,
                                        file_size
                                    ))
                        elif isinstance(artifacts, list):  # list of files
                            for i, artifact_path in enumerate(artifacts):
                                if Path(artifact_path).exists():
                                    file_size = Path(artifact_path).stat().st_size
                                    conn.execute("""
                                        INSERT OR REPLACE INTO artifacts 
                                        (experiment_id, artifact_name, artifact_path, artifact_type, file_size)
                                        VALUES (?, ?, ?, ?, ?)
                                    """, (
                                        experiment.experiment_id,
                                        f"{artifact_type}_{i}",
                                        artifact_path,
                                        artifact_type,
                                        file_size
                                    ))
                
                conn.commit()
                self.logger.info(f"Saved experiment {experiment.experiment_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save experiment {experiment.experiment_id}: {e}")
            return False
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Load experiment record from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load main experiment data
                exp_row = conn.execute("""
                    SELECT * FROM experiments WHERE experiment_id = ?
                """, (experiment_id,)).fetchone()
                
                if not exp_row:
                    return None
                
                # Load parameters
                param_rows = conn.execute("""
                    SELECT param_name, param_value, param_type FROM parameters 
                    WHERE experiment_id = ?
                """, (experiment_id,)).fetchall()
                
                parameters = {}
                for row in param_rows:
                    parameters[row['param_name']] = json.loads(row['param_value'])
                
                # Load metrics
                metric_rows = conn.execute("""
                    SELECT metric_name, metric_value, is_primary FROM metrics 
                    WHERE experiment_id = ?
                """, (experiment_id,)).fetchall()
                
                metrics_data = {}
                additional_metrics = {}
                
                for row in metric_rows:
                    if row['is_primary']:
                        metrics_data['primary_metric'] = row['metric_value']
                        metrics_data['primary_metric_name'] = row['metric_name']
                    else:
                        if row['metric_name'] in ['cv_mean', 'cv_std', 'train_time', 'predict_time']:
                            metrics_data[row['metric_name']] = row['metric_value']
                        else:
                            additional_metrics[row['metric_name']] = row['metric_value']
                
                metrics_data['additional_metrics'] = additional_metrics
                
                # Load artifacts
                artifact_rows = conn.execute("""
                    SELECT artifact_name, artifact_path, artifact_type FROM artifacts 
                    WHERE experiment_id = ?
                """, (experiment_id,)).fetchall()
                
                artifacts_data = {
                    'model_path': None,
                    'plots': {},
                    'data_files': {},
                    'logs': []
                }
                
                for row in artifact_rows:
                    if row['artifact_type'] == 'model_path':
                        artifacts_data['model_path'] = row['artifact_path']
                    elif row['artifact_type'] == 'plots':
                        artifacts_data['plots'][row['artifact_name']] = row['artifact_path']
                    elif row['artifact_type'] == 'data_files':
                        artifacts_data['data_files'][row['artifact_name']] = row['artifact_path']
                    elif row['artifact_type'] == 'logs':
                        artifacts_data['logs'].append(row['artifact_path'])
                
                # Reconstruct experiment
                config = ExperimentConfig(
                    experiment_name=exp_row['experiment_name'],
                    description=exp_row['description'] or "",
                    tags=json.loads(exp_row['tags'] or "[]"),
                    dataset_name=exp_row['dataset_name'] or "",
                    dataset_hash=exp_row['dataset_hash'] or "",
                    target_column=exp_row['target_column'] or "",
                    task_type=exp_row['task_type'] or ""
                )
                
                metrics = ExperimentMetrics(**metrics_data)
                artifacts = ExperimentArtifacts(**artifacts_data)
                
                experiment = ExperimentRecord(
                    experiment_id=experiment_id,
                    config=config,
                    parameters=parameters,
                    metrics=metrics,
                    artifacts=artifacts,
                    timestamp=datetime.fromisoformat(exp_row['timestamp']),
                    duration=exp_row['duration'] or 0.0,
                    status=exp_row['status'] or "completed",
                    notes=exp_row['notes'] or ""
                )
                
                return experiment
                
        except Exception as e:
            self.logger.error(f"Failed to load experiment {experiment_id}: {e}")
            return None
    
    def list_experiments(self, limit: int = 50, offset: int = 0, 
                        filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = """
                    SELECT experiment_id, experiment_name, description, task_type, 
                           timestamp, duration, status, dataset_name, target_column
                    FROM experiments 
                """
                params = []
                
                if filters:
                    where_conditions = []
                    if 'task_type' in filters:
                        where_conditions.append("task_type = ?")
                        params.append(filters['task_type'])
                    if 'dataset_name' in filters:
                        where_conditions.append("dataset_name = ?")
                        params.append(filters['dataset_name'])
                    if 'status' in filters:
                        where_conditions.append("status = ?")
                        params.append(filters['status'])
                    if 'experiment_name' in filters:
                        where_conditions.append("experiment_name LIKE ?")
                        params.append(f"%{filters['experiment_name']}%")
                    
                    if where_conditions:
                        query += " WHERE " + " AND ".join(where_conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                rows = conn.execute(query, params).fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            return []
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment and associated artifacts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete from all tables
                conn.execute("DELETE FROM artifacts WHERE experiment_id = ?", (experiment_id,))
                conn.execute("DELETE FROM metrics WHERE experiment_id = ?", (experiment_id,))
                conn.execute("DELETE FROM parameters WHERE experiment_id = ?", (experiment_id,))
                conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                conn.commit()
                
                self.logger.info(f"Deleted experiment {experiment_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False

class ExperimentTracker:
    """
    Main experiment tracking interface.
    
    Features:
    - Automatic experiment logging
    - Context management for active experiments
    - Artifact management and storage
    - Integration with ML frameworks
    """
    
    def __init__(self, experiment_dir: str = "experiments", db_path: str = None):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory for storing experiment artifacts
            db_path: Path to SQLite database (default: experiment_dir/experiments.db)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if db_path is None:
            db_path = self.experiment_dir / "experiments.db"
        
        self.db = ExperimentDatabase(str(db_path))
        self.logger = logging.getLogger(__name__)
        
        self.current_experiment: Optional[ExperimentRecord] = None
        self.start_time: Optional[datetime] = None
    
    def start_experiment(self, config: ExperimentConfig, parameters: Dict[str, Any]) -> str:
        """
        Start a new experiment.
        
        Args:
            config: Experiment configuration
            parameters: Model/training parameters
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        self.start_time = datetime.now(timezone.utc)
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment record
        metrics = ExperimentMetrics(primary_metric=0.0, primary_metric_name="unknown")
        artifacts = ExperimentArtifacts()
        
        self.current_experiment = ExperimentRecord(
            experiment_id=experiment_id,
            config=config,
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts,
            timestamp=self.start_time,
            status="running"
        )
        
        self.logger.info(f"Started experiment {experiment_id}: {config.experiment_name}")
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], primary_metric: str = None):
        """Log experiment metrics"""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        # Update metrics
        for name, value in metrics.items():
            if primary_metric and name == primary_metric:
                self.current_experiment.metrics.primary_metric = value
                self.current_experiment.metrics.primary_metric_name = name
            else:
                self.current_experiment.metrics.additional_metrics[name] = value
        
        self.logger.info(f"Logged metrics for experiment {self.current_experiment.experiment_id}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None, 
                    artifact_type: str = "data") -> str:
        """
        Log an artifact (file) for the experiment.
        
        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name for the artifact (defaults to filename)
            artifact_type: Type of artifact (model, plot, data, log)
            
        Returns:
            Path where artifact was stored
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        
        if artifact_name is None:
            artifact_name = artifact_path.name
        
        # Copy artifact to experiment directory
        exp_dir = self.experiment_dir / self.current_experiment.experiment_id
        stored_path = exp_dir / artifact_name
        shutil.copy2(artifact_path, stored_path)
        
        # Update artifacts
        if artifact_type == "model":
            self.current_experiment.artifacts.model_path = str(stored_path)
        elif artifact_type == "plot":
            self.current_experiment.artifacts.plots[artifact_name] = str(stored_path)
        elif artifact_type == "data":
            self.current_experiment.artifacts.data_files[artifact_name] = str(stored_path)
        elif artifact_type == "log":
            self.current_experiment.artifacts.logs.append(str(stored_path))
        
        self.logger.info(f"Logged artifact {artifact_name} for experiment {self.current_experiment.experiment_id}")
        return str(stored_path)
    
    def save_model(self, model, model_name: str = "model.pkl") -> str:
        """Save trained model as experiment artifact"""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        exp_dir = self.experiment_dir / self.current_experiment.experiment_id
        model_path = exp_dir / model_name
        
        # Save model using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.current_experiment.artifacts.model_path = str(model_path)
        self.logger.info(f"Saved model for experiment {self.current_experiment.experiment_id}")
        return str(model_path)
    
    def end_experiment(self, status: str = "completed", notes: str = "") -> bool:
        """
        End current experiment and save to database.
        
        Args:
            status: Final status (completed, failed)
            notes: Optional notes about the experiment
            
        Returns:
            True if successful
        """
        if not self.current_experiment:
            raise ValueError("No active experiment to end.")
        
        # Calculate duration
        if self.start_time:
            duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            self.current_experiment.duration = duration
        
        self.current_experiment.status = status
        self.current_experiment.notes = notes
        
        # Save to database
        success = self.db.save_experiment(self.current_experiment)
        
        if success:
            self.logger.info(f"Ended experiment {self.current_experiment.experiment_id} with status: {status}")
        
        # Clean up
        self.current_experiment = None
        self.start_time = None
        
        return success
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve experiment by ID"""
        return self.db.load_experiment(experiment_id)
    
    def list_experiments(self, limit: int = 50, **filters) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        return self.db.list_experiments(limit=limit, filters=filters)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment and its artifacts"""
        # Delete from database
        success = self.db.delete_experiment(experiment_id)
        
        # Delete artifacts directory
        exp_dir = self.experiment_dir / experiment_id
        if exp_dir.exists():
            try:
                shutil.rmtree(exp_dir)
                self.logger.info(f"Deleted experiment directory: {exp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to delete experiment directory {exp_dir}: {e}")
        
        return success
    
    def calculate_dataset_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataset for tracking"""
        # Create hash based on shape and sample of data
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        columns_str = ",".join(df.columns.tolist())
        
        # Sample some values for hash (to handle large datasets efficiently)
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        sample_str = str(sample_df.values.tobytes())
        
        combined_str = f"{shape_str}|{columns_str}|{sample_str}"
        return hashlib.md5(combined_str.encode()).hexdigest()[:16]