import logging
import time
import os
import socketio
import numpy as np
from typing import Dict
import threading

# --- MODIFICA CHIAVE: Import corretto ---
from model_manager import ModelManager
from utils import object_to_pickle_string, pickle_string_to_object, encrypt_weights, decrypt_weights


class ContextFilter(logging.Filter):
    """A logging filter to add client_id context to log records."""

    def __init__(self, client_id: str):
        super().__init__()
        self.client_id = client_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.client_id = self.client_id
        return True


class FederatedClient:
    """
    Implements the client-side logic for federated learning.
    """

    def __init__(self, config: Dict, dataset_path: str):
        # --- MODIFICA CHIAVE: Il tipo è ora ModelManager ---
        self.local_model: ModelManager = None
        self.config: Dict = config
        self.dataset_path: str = dataset_path
        self.client_id: str = os.path.basename(dataset_path)
        self.logger: logging.LoggerAdapter = self._setup_logger()

        self.encryption_mode: str = self.config.get('encryption_mode', 'none')
        if self.encryption_mode != 'no_encryption':
            self.logger.info("Encryption enabled. Keys will be requested from the Trusted Authority.")
            self.paillier_pubkey = None
            self.paillier_privkey = None
            self.keys_received_event = threading.Event()

        self.sio = socketio.Client(logger=True, request_timeout=10, reconnection=True)
        self._register_event_handlers()
        self.connect_to_server()

    def _setup_logger(self) -> logging.LoggerAdapter:
        worker_id = self.config.get('worker_id', 'N/A')
        logger_name = f"FederatedClient-W{worker_id}"
        base_logger = logging.getLogger(logger_name)

        if not base_logger.hasHandlers():
            datestr = time.strftime('%d%m')
            timestr = time.strftime('%m%d%H%M')
            log_dir_base = self.config.get('log_dir', 'logs')
            log_dir = os.path.join(log_dir_base, datestr, "FL-Client-LOG")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f'{timestr}_{self.client_id}.log'))
            file_handler.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.WARN)
            formatter = logging.Formatter('%(asctime)s - %(client_id)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            base_logger.setLevel(logging.INFO)
            base_logger.addHandler(file_handler)
            base_logger.addHandler(stream_handler)

        adapter = logging.LoggerAdapter(base_logger, {'client_id': self.client_id})
        log_filter = ContextFilter(self.client_id)
        if not any(isinstance(f, ContextFilter) for f in adapter.logger.filters):
            adapter.logger.addFilter(log_filter)
        return adapter

    def _process_server_weights(self, data: Dict):
        """
        Processes weights from the server: decrypts if needed, then rescales.

        The server sends a weighted *sum* and the client turns it into the
        model, because under Paillier the server can only add ciphertexts and
        scale them by plaintext numbers - the division has to happen after
        decryption, here.

        What to divide by depends on the aggregation rule, so the server now
        says it explicitly in `aggregation_denominator`:
          - FedAvg / FedProx / FedLC: N, the total training size, turning
            `sum_k n_k * W_k` into the average.
          - FedDisco / FIPA: 1.0, because the server's output is already the
            finished model and dividing again would shrink it.

        The fallback to `total_training_size` keeps this working against a
        server that has not been updated. A denominator of 0 means round 0,
        whose payload is the initial weights rather than a sum: using them
        unscaled is correct.
        """
        weights_pickled = data['current_weights']
        denominator = data.get('aggregation_denominator',
                               data.get('total_training_size', 0))
        weights_data = pickle_string_to_object(weights_pickled)
        plaintext_sum = None

        first_element = weights_data[0]
        if isinstance(first_element, dict):
            if self.encryption_mode == 'no_encryption':
                self.logger.error("Received encrypted-format weights while in 'no_encryption' mode. Mismatch.")
                return None
            try:
                self.logger.info("Data is in dictionary format, attempting decryption.")
                plaintext_sum = decrypt_weights(self.paillier_privkey, weights_data, logger=self.logger)
            except Exception as e:
                self.logger.error("Error during weight decryption: %s", e, exc_info=True)
                return None
        elif isinstance(first_element, np.ndarray):
            # Formato in chiaro: List[np.ndarray]
            self.logger.info("Data is in numpy format, assuming plaintext.")
            plaintext_sum = weights_data
        else:
            self.logger.error(f"Unknown weights format received. Type: {type(first_element)}")
            return None
        # --- FINE MODIFICA ---
        if denominator > 0:
            self.logger.info("Rescaling server payload by denominator %.4f.", denominator)
            return [w / denominator for w in plaintext_sum]
        else:
            self.logger.warning("Aggregation denominator is 0. Using weights as is.")
            return plaintext_sum

    def connect_to_server(self) -> None:
        server_address = "http://" + self.config['ip_address'] + ":" + str(self.config['port'])
        self.logger.info("Connecting to server at %s", server_address)
        try:
            self.sio.connect(server_address, transports=['websocket'])
            self.logger.info("Sending wake up message to the server.")
            self.sio.emit('client_wake_up')
            self.sio.wait()
        except socketio.exceptions.ConnectionError as e:
            self.logger.error("Failed to connect to the server: %s", e)
            exit(1)

    def _register_event_handlers(self) -> None:
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        self.sio.on('reconnect', self._on_reconnect)
        self.sio.on('shutdown', self._on_shutdown)
        self.sio.on('init', self._on_init)
        self.sio.on('request_update', self._on_request_update)
        self.sio.on('stop_and_eval', self._on_stop_and_eval)
        self.sio.on('distribute_calibration', self._on_distribute_calibration)

    def _on_connect(self):
        self.logger.info("Successfully connected with SID: %s", self.sio.sid)

    def _on_disconnect(self):
        self.logger.info("Disconnected from the server."); self.sio.disconnect()

    def _on_reconnect(self):
        self.logger.info("Reconnected to the server.")

    def _on_shutdown(self):
        self.logger.info("Received shutdown signal."); self.sio.disconnect()

    def _on_init(self, data: Dict):
        self.logger.info("Received initialization signal from server.")
        if self.encryption_mode != 'no_encryption':
            ta_address = data['ta_address']
            self.logger.info("Connecting to Trusted Authority at %s.", ta_address)
            ta_sio = socketio.Client(reconnection=False)

            @ta_sio.on('distribute_keys')
            def on_receive_keys(key_data):
                self.logger.info("Keys received from Trusted Authority.")
                self.paillier_pubkey = pickle_string_to_object(key_data['pubkey'])
                self.paillier_privkey = pickle_string_to_object(key_data['privkey'])
                self.keys_received_event.set()
                ta_sio.disconnect()

            try:
                ta_sio.connect(ta_address, transports=['websocket'])
                ta_sio.emit('request_keys')
                if not self.keys_received_event.wait(timeout=30):
                    self.logger.error("CRITICAL: Did not receive keys from TA within timeout.")
                    exit(1)
            except Exception as e:
                self.logger.error("CRITICAL: Failed to get keys from TA: %s", e)
                exit(1)

        self._initialize_model_and_report_ready()

    def _initialize_model_and_report_ready(self):
        self.logger.info("Initializing local model.")

        # --- MODIFICA CHIAVE: Istanza del nuovo ModelManager ---
        self.local_model = ModelManager(
            config=self.config,
            dataset_path=self.dataset_path
        )

        self.logger.info("Local model initialized. Calculating data stats...")
        samples_per_class = self.local_model.get_samples_per_class()
        self.logger.info("Stats calculated. Sending 'client_ready' to the main server.")
        # `client_id` is the name of this client's data directory (`client_0`).
        # Unlike the Socket.IO session id it survives a reconnect, so the server
        # can key the label distributions by it and pair them with the weight
        # updates that arrive later, in a different order.
        self.sio.emit('client_ready', {
            'client_id': self.client_id,
            'samples_per_class': object_to_pickle_string(samples_per_class),
        })

    def _on_distribute_calibration(self, data: Dict):
        self.logger.info("Received static calibration term from the server.")
        calibration_val = pickle_string_to_object(data['calibration_val'])
        self.local_model.set_calibration_term(calibration_val)

    def _on_request_update(self, data: Dict):
        self.logger.info("Received model update request for round %s. Starting worker thread.", data['round_number'])
        worker_thread = threading.Thread(target=self._update_worker, args=(data,))
        worker_thread.daemon = True
        worker_thread.start()

    def _update_worker(self, data: Dict):
        try:
            self.logger.info("Worker thread started for round %s.", data['round_number'])
            averaged_weights = self._process_server_weights(data)
            if averaged_weights is None:
                self.logger.error("Worker thread: Failed to obtain valid weights.")
                return

            self.local_model.set_weights(averaged_weights)

            _, train_map, train_loss, train_size = self.local_model.train(
                epochs=data['epochs'], lr=data['learning_rate'], batch_size=data['batch_size'],
                algorithm=self.config.get("aggregation_algorithm", "FedAvg"),
                global_weights=averaged_weights, mu=self.config.get("fedprox_mu", 0.0)
            )
            local_weights = self.local_model.get_weights()

            if self.encryption_mode != 'no_encryption':
                weights_to_send = encrypt_weights(self.paillier_pubkey, local_weights,
                                                  encryption_mode=self.encryption_mode, logger=self.logger)
            else:
                weights_to_send = local_weights

            response = {
                'client_id': self.client_id,
                'round_number': data['round_number'], 'train_loss': train_loss,
                'avg_f1': np.mean(list(train_map['f1_score'])) if isinstance(train_map['f1_score'], list) else
                train_map['f1_score'],
                'avg_acc': np.mean(list(train_map['accuracy'])) if isinstance(train_map['accuracy'], list) else
                train_map['accuracy'],
                'train_size': train_size, 'weights': object_to_pickle_string(weights_to_send)
            }
            self.logger.info("Worker thread: Sending client update for round %s.", data['round_number'])
            self.sio.emit('client_update', response)
            self.logger.info("--- Worker thread Round %s Training Summary ---", data['round_number'])
            self.logger.info("Client Train Loss: %.4f", train_loss)
            self.logger.info("-------------------------------------------")

        except Exception as e:
            self.logger.error("An error occurred in the update worker thread: %s", e, exc_info=True)

    def _on_stop_and_eval(self, data: Dict):
        self.logger.info("Received final aggregated model for evaluation.")
        final_weights = self._process_server_weights(data)
        if final_weights is None:
            self.logger.error("Evaluation failed: Could not process server weights.")
            return

        self.local_model.set_weights(final_weights)
        self.logger.info("Evaluating the final model.")
        valid_loss, metric_score, _, test_size = self.local_model.validate(data['batch_size'])
        response = {
            'test_loss': valid_loss, 'test_f1': metric_score['f1_score'],
            'test_acc': metric_score['accuracy'], 'test_prec': metric_score['precision'],
            'test_recall': metric_score['recall'], 'test_size': test_size,
        }
        self.logger.info("Sending final evaluation to the server.")
        self.sio.emit('client_eval', response)

        if data.get('STOP', False):
            self.logger.info("Federated training finished. Shutting down client.")
            exit(0)