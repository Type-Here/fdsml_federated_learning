import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict
import logging

# --- MODIFICA CHIAVE: Import corretto ---
from model_manager import ModelManager
from utils import sum_encrypted_weights, multiply_encrypted_weights_by_scalar


class Aggregator:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.current_weights = self._initialize_weights()
        self.metrics_history: Dict[int, Dict] = {}
        self.best_f1_score: float = 0.0
        self.best_accuracy: float = 0.0
        self.best_precision: float = 0.0
        self.best_recall: float = 0.0
        self.best_loss: float = float('inf')
        self.best_model_weights: List[np.ndarray] = None
        self.best_round: int = -1
        self.prev_test_loss: float = None
        self.early_stop_counter: int = 0
        self.training_start_time: float = time.time()
        self.run_summary: Dict = None

    def _initialize_weights(self) -> List[np.ndarray]:
        """
        Initializes the model weights by creating a temporary model instance.
        """
        self.logger.info("Initializing model weights...")

        # --- MODIFICA CHIAVE: Istanza del nuovo ModelManager ---
        temp_model = ModelManager(config=self.config, dataset_path="")

        initial_weights = temp_model.get_weights()
        self.logger.info(f"Trainable parameters initialized for model {self.config['model_name']}.")
        del temp_model
        return initial_weights

    def aggregate_weights(self, client_updates: List[Dict], algorithm: str) -> bool:
        """
        Aggregates weights from clients by performing a weighted SUMMATION.
        The final division (averaging) is handled by the clients.
        """
        self.logger.info(f"Aggregating weights using '{algorithm}'...")
        if not client_updates:
            self.logger.warning("No client updates received for aggregation.")
            return False

        client_weights = [u['weights'] for u in client_updates]
        client_sizes = [u['train_size'] for u in client_updates]

        if algorithm in ["FedAvg", "FedProx", "FedLC"]:
            total_size = sum(client_sizes)
            if total_size == 0:
                self.logger.warning("Total training size is 0. Skipping aggregation.")
                return False

            # Perform weighted summation
            summed_weights = [w * client_sizes[0] for w in client_weights[0]]
            for client_idx in range(1, len(client_weights)):
                for i in range(len(summed_weights)):
                    summed_weights[i] += client_weights[client_idx][i] * client_sizes[client_idx]

            self.current_weights = summed_weights
            self.logger.info(f"Successfully performed weighted summation. Total training size: {total_size}.")
            return True
        else:
            raise ValueError(f"Aggregation algorithm '{algorithm}' is not supported.")

    def aggregate_encrypted_updates(self, round_client_updates):
        client_sizes = [update['train_size'] for update in round_client_updates]
        summed_encrypted_weights = multiply_encrypted_weights_by_scalar(
            round_client_updates[0]['weights'], client_sizes[0]
        )
        for i in range(1, len(round_client_updates)):
            weighted_update = multiply_encrypted_weights_by_scalar(
                round_client_updates[i]['weights'], client_sizes[i]
            )
            summed_encrypted_weights = sum_encrypted_weights(summed_encrypted_weights, weighted_update)
        self.current_weights = summed_encrypted_weights
        self.logger.info("Encrypted aggregation (weighted sum) complete.")

    def aggregate_train_loss_weighted(self, client_losses: List[float], client_sizes: List[int],
                                      current_round: int) -> None:
        total_size = sum(client_sizes)
        if total_size == 0: return
        weighted_loss = np.average(client_losses, weights=client_sizes)
        self.logger.info(f"Round {current_round} - Weighted average training loss: {weighted_loss:.4f}")
        if current_round not in self.metrics_history: self.metrics_history[current_round] = {}
        self.metrics_history[current_round]['train_loss'] = weighted_loss

    def aggregate_evaluation_results(self, eval_updates: List[Dict], current_round: int) -> bool:
        client_sizes = [u['test_size'] for u in eval_updates]
        total_test_size = sum(client_sizes)
        if total_test_size == 0: return False

        avg_test_loss = np.average([u['test_loss'] for u in eval_updates], weights=client_sizes)
        avg_test_f1 = np.average([u['test_f1'] for u in eval_updates], weights=client_sizes)
        avg_test_acc = np.average([u['test_acc'] for u in eval_updates], weights=client_sizes)
        avg_test_prec = np.average([u['test_prec'] for u in eval_updates], weights=client_sizes)
        avg_test_recall = np.average([u['test_recall'] for u in eval_updates], weights=client_sizes)

        self.logger.info("--- Round %d Evaluation Summary ---", current_round)
        self.logger.info("Average Test Loss: %.4f, F1 Score: %.4f, Accuracy: %.4f", avg_test_loss, avg_test_f1,
                         avg_test_acc)

        self.metrics_history[current_round].update({
            'test_loss': avg_test_loss, 'test_f1': avg_test_f1, 'test_acc': avg_test_acc,
            'test_prec': avg_test_prec, 'test_recall': avg_test_recall
        })

        if avg_test_f1 > self.best_f1_score:
            self.best_f1_score = avg_test_f1
            self.best_accuracy = avg_test_acc
            self.best_precision = avg_test_prec
            self.best_recall = avg_test_recall
            self.best_loss = avg_test_loss
            self.best_model_weights = self.current_weights
            self.best_round = current_round

        if self.prev_test_loss is not None and avg_test_loss > self.prev_test_loss:
            self.early_stop_counter += 1
        else:
            self.early_stop_counter = 0
        self.prev_test_loss = avg_test_loss

        return self.early_stop_counter >= self.config['early_stop_patience']

    def log_best_model_stats(self):
        self.logger.info("=" * 30)
        self.logger.info("Federated Training Finished")
        self.logger.info("Best model at round: %d | F1: %.4f | Acc: %.4f | Loss: %.4f",
                         self.best_round, self.best_f1_score, self.best_accuracy, self.best_loss)
        self.logger.info("=" * 30)

    def get_stats(self) -> pd.DataFrame:
        if not self.metrics_history: return pd.DataFrame()
        return pd.DataFrame.from_dict(self.metrics_history, orient='index')

    def save_results(self) -> None:
        metrics_dir = self.config['run_metrics_output_path']
        os.makedirs(metrics_dir, exist_ok=True)
        timestr = time.strftime('%Y%m%d-%H%M%S')
        metrics_df = self.get_stats()
        worker_id = self.config.get('worker_id', 'NA')
        dataset_name = self.config.get('dataset_name', 'unknown_dataset')
        metrics_filename = os.path.join(metrics_dir, f"{timestr}_{dataset_name}_worker_{worker_id}.csv")
        metrics_df.to_csv(metrics_filename)
        self.logger.info("Saved training metrics to %s", metrics_filename)
        self.run_summary = self.config.copy()
        self.run_summary.update({
            'best_round': self.best_round, 'best_f1': self.best_f1_score,
            'best_acc': self.best_accuracy, 'best_prec': self.best_precision,
            'best_recall': self.best_recall, 'best_loss': self.best_loss,
            'total_duration': time.time() - self.training_start_time,
            'dataframe_path': metrics_filename,
        })
        keys_to_remove = ['shared_csv_path', 'run_metrics_output_path', 'base_csv_path',
                          'base_log_path', 'base_plot_path', 'base_split_data_path',
                          'ip_address', 'port', 'ta_port']
        for key in keys_to_remove:
            self.run_summary.pop(key, None)
        self.logger.info("Run summary has been prepared.")

    def get_run_summary(self) -> Dict:
        return self.run_summary

    def get_parameter_plots(self) -> List[str]:
        df = self.get_stats()
        if df.empty: return []
        plots_dir = self.config.get('plot_dir', "static/plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_paths = []
        timestr = time.strftime('%Y%m%d-%H%M%S')
        plt.figure()
        plt.title("Train and Test Loss vs. Rounds")
        plt.plot(df.index, df['train_loss'], label="Train Loss")
        plt.plot(df.index, df['test_loss'], label="Test Loss")
        plt.xlabel("Round");
        plt.ylabel("Loss");
        plt.legend()
        path = os.path.join(plots_dir, f"{timestr}_Loss.png")
        plt.savefig(path);
        plt.close()
        plot_paths.append(path)
        plt.figure()
        plt.title("Test Metrics vs. Rounds")
        plt.plot(df.index, df['test_f1'], label="F1 Score")
        plt.plot(df.index, df['test_acc'], label="Accuracy")
        plt.xlabel("Round");
        plt.ylabel("Score");
        plt.legend()
        path = os.path.join(plots_dir, f"{timestr}_Metrics.png")
        plt.savefig(path);
        plt.close()
        plot_paths.append(path)
        return plot_paths