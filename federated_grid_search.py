import json
import sys
import threading
import time
import itertools
import requests
from requests.exceptions import ConnectionError
import socketio

import PCNAME
import csv
import os
from typing import Dict, List, Any, Generator, Set, FrozenSet, Tuple
import multiprocessing
import queue

from config_fingerprint import (
    EXTRA_FINGERPRINT_KEYS,
    get_config_fingerprint,
    normalize_augmentation_keys,
    normalize_fipa_keys,
    normalize_partition_keys,
)
from trusted_authority import TrustedAuthority
from seeding import seed_from_config
import federated_server
import run_multiple_clients

# Default number of federated simulations run at the same time. Sized for a
# multi-GPU machine; override with "num_parallel_executions" in the config file
# on a single-GPU box or on Colab, where 12 concurrent runs exhaust memory.
DEFAULT_NUM_PARALLEL_EXECUTIONS = 12
GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'
VERBOSE_DUPLICATE_CHECK = False

def wait_for_server_ready(url, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200: return True
        except ConnectionError:
            time.sleep(1)
    return False


def load_json(filename: str) -> Dict:
    with open(filename) as f: return json.load(f)


def append_results_to_csv(csv_path: str, run_summary: Dict, lock: multiprocessing.Lock):
    lock.acquire()
    try:
        all_keys = set(run_summary.keys())
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                all_keys.update(header)

        file_is_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if file_is_empty:
                writer.writeheader()
            writer.writerow(run_summary)
    finally:
        lock.release()


def generate_configurations(base_config: Dict, search_space: Dict) -> Generator[Dict[str, Any], None, None]:
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def start_server_thread(config: Dict, server_instance_ref: List) -> None:
    worker_id = config.get('worker_id', 'N/A');
    port = config['port']
    print(f"[Worker {worker_id}] Starting server thread on port {port}...")
    try:
        server = federated_server.FederatedServer(config);
        server_instance_ref.append(server);
        server.run()
    except Exception as e:
        print(f"!!!!!!!!!! [Worker {worker_id}] SERVER THREAD CRASHED on port {port} !!!!!!!!!!!")
        print(f"ERROR: {e}");
        import traceback;
        traceback.print_exc()
    print(f"[Worker {worker_id}] Server thread has finished.")


def start_ta_thread(config: Dict, ta_instance_ref: List) -> None:
    print(f"[Worker {config.get('worker_id', 'N/A')}] Starting TA thread on port {config['ta_port']}...")
    ta = TrustedAuthority(host=config['ip_address'], port=config['ta_port']);
    ta_instance_ref.append(ta);
    ta.run()
    print(f"[Worker {config.get('worker_id', 'N/A')}] TA thread has finished.")


def run_grid_search_worker(
        worker_id: int,
        task_queue: multiprocessing.JoinableQueue,
        base_port: int,
        csv_lock: multiprocessing.Lock
) -> None:
    # ... (invariato)
    print(f"--- Starting Grid Search Worker {worker_id} ---")
    while True:
        try:
            config = task_queue.get()
            if config is None:
                print(f"--- Worker {worker_id} received sentinel. Shutting down. ---")
                break
            dataset_name = config['dataset_name']
            run_identifier = f"{dataset_name}_run_{worker_id}_{config['model_name']}"
            pc_name = PCNAME.name
            worker_splitting_dir = os.path.join(config['base_split_data_path']+f'_{pc_name}', run_identifier)
            worker_log_dir = os.path.join(config['base_log_path']+f'_{pc_name}', run_identifier)
            worker_plot_dir = os.path.join(config['base_plot_path']+f'_{pc_name}', run_identifier)
            worker_metrics_dir = os.path.join(config['base_csv_path']+f'_{pc_name}', "runs")
            # Not per run_identifier like the logs: checkpoints from every worker
            # collect in one directory, because what comes next reads them as a
            # set - one model per configuration, compared side by side.
            worker_checkpoint_dir = config.get('base_checkpoint_path', 'checkpoints') + f'_{pc_name}'
            os.makedirs(worker_splitting_dir, exist_ok=True)
            os.makedirs(worker_log_dir, exist_ok=True)
            os.makedirs(worker_plot_dir, exist_ok=True)
            os.makedirs(worker_metrics_dir, exist_ok=True)
            os.makedirs(worker_checkpoint_dir, exist_ok=True)
            print("\n" + "#" * 80)
            print(f"### [Worker {worker_id}] DEQUEUED NEW CONFIG FOR: {dataset_name} | {config['model_name']} ###")
            config['worker_id'] = worker_id
            # Seed once per configuration, not once per worker: the result of a
            # config must not depend on which worker dequeued it or in what
            # order. Without this, random.sample in the server picks different
            # clients every time and two runs of the same config diverge.
            applied_seed = seed_from_config(config)
            config['seed'] = applied_seed
            print(f"[Worker {worker_id}] Global seed set to {applied_seed}.")
            config['port'] = base_port + worker_id * 2
            config['ta_port'] = base_port + worker_id * 2 + 1
            config['splitting_dir'] = worker_splitting_dir
            config['log_dir'] = worker_log_dir
            config['plot_dir'] = worker_plot_dir
            config['run_metrics_output_path'] = worker_metrics_dir
            config['run_checkpoint_output_path'] = worker_checkpoint_dir
            config["MIN_NUM_WORKERS"] = int(config['num_clients'] * config['models_percentage'])
            ta_instance_ref = [];
            server_instance_ref = []
            use_encryption = config.get('encryption_mode', 'no_encryption') != 'no_encryption'
            ta_thread = None
            if use_encryption:
                ta_thread = threading.Thread(target=start_ta_thread, args=(config, ta_instance_ref))
                ta_thread.start()
                time.sleep(3)
            server_thread = threading.Thread(target=start_server_thread, args=(config, server_instance_ref));
            server_thread.start()
            server_url = f"http://{config['ip_address']}:{config['port']}"
            if wait_for_server_ready(server_url):
                run_multiple_clients.main(config)
            else:
                print(f"[Worker {worker_id}] Server failed to start. Skipping.")
            server_thread.join()
            if use_encryption and ta_thread is not None:
                try:
                    sio_client = socketio.Client(reconnection=False)
                    sio_client.connect(f"http://{config['ip_address']}:{config['ta_port']}", transports=['websocket'])
                    sio_client.emit('shutdown_ta');
                    time.sleep(1);
                    sio_client.disconnect()
                except Exception as e:
                    print(f"[Worker {worker_id}] Could not shutdown TA: {e}")
                ta_thread.join(timeout=15)
            if server_instance_ref:
                run_summary = server_instance_ref[0].aggregator.get_run_summary()
                if run_summary:
                    append_results_to_csv(config['shared_csv_path'], run_summary, csv_lock)
            time.sleep(1)
        except queue.Empty:
            continue
        finally:
            task_queue.task_done()


def main(config_path: str = GRID_SEARCH_CONFIG_PATH):
    """Run the grid described by `config_path`.

    The path is a parameter so the smoke configuration can be run without
    editing this file, which matters on Colab where the repo is a fresh clone:
    `python federated_grid_search.py smoke_config.json`.
    """
    print(f"Loading grid configuration from '{config_path}'.")
    base_grid_config = load_json(config_path)
    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()
    pc_name = PCNAME.name
    base_csv_path = base_grid_config['base_csv_path']+f'_{pc_name}'
    os.makedirs(base_csv_path, exist_ok=True)
    os.makedirs(base_grid_config.get('base_log_path', 'logs')+f'_{pc_name}', exist_ok=True)
    os.makedirs(base_grid_config.get('base_plot_path', 'static/plots')+f'_{pc_name}', exist_ok=True)
    os.makedirs(base_grid_config.get('base_split_data_path', f'run_dataset')+f'_{pc_name}', exist_ok=True)
    path_risultati = os.path.join(pc_name, f'federated_grid_search_results_{pc_name}.csv')
    shared_csv_path = os.path.join(base_csv_path, path_risultati)

    directory_path = os.path.dirname(shared_csv_path)

    os.makedirs(directory_path, exist_ok=True)

    executed_fingerprints: Set[FrozenSet[Tuple[str, str]]] = set()
    common_search_space = base_grid_config['common_search_space']
    model_specific_search_space = base_grid_config['model_specific_search_space']

    all_possible_search_keys = set(common_search_space.keys())
    for model_params in model_specific_search_space.values():
        all_possible_search_keys.update(model_params.keys())
    # The partitioning keys are listed explicitly so they count towards the
    # fingerprint even when a config declares them as fixed parameters rather
    # than as search axes. Without this, two runs differing only in
    # `dirichlet_alpha` would look identical and the second would be skipped.
    all_possible_search_keys.update(EXTRA_FINGERPRINT_KEYS)
    # NOTE: `seed` is deliberately NOT fingerprinted. Adding it would make every
    # row the lab already produced (which has no seed column) differ from a new
    # run, re-queueing the entire previous grid. If we ever want seed-variance
    # runs, that needs a backfill of the default onto the CSV rows first.

    if os.path.exists(shared_csv_path) and os.path.getsize(shared_csv_path) > 0:
        with open(shared_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalizza la riga del CSV nello stesso modo in cui normalizzeremo le nuove config
                model_name_from_row = row.get('model_name', '')
                if row.get('aggregation_algorithm') != "FedProx": row['fedprox_mu'] = '0.0'
                normalize_partition_keys(row)
                normalize_fipa_keys(row)
                normalize_augmentation_keys(row)
                if 'ResNet' in model_name_from_row or 'GoogLeNet' in model_name_from_row or 'AlexNet' in model_name_from_row:
                    row.setdefault('image_size', '224')
                    row.setdefault('convnet_hidden1', '-1')
                    row.setdefault('convnet_hidden2', '-1')
                elif 'ConvNet' in model_name_from_row:
                    row.setdefault('num_custom_layers', '-1')

                # Usa il set di chiavi globale per creare il fingerprint
                fingerprint = get_config_fingerprint(row, all_possible_search_keys)
                executed_fingerprints.add(fingerprint)
    print(f"Loaded {len(executed_fingerprints)} fingerprints of previously executed configurations.")

    fixed_params = {k: v for k, v in base_grid_config.items() if
                    k not in ['datasets', 'common_search_space', 'model_specific_search_space']}
    fixed_params['shared_csv_path'] = shared_csv_path
    configs_to_run_count = 0
    total_generated_configs = 0

    for dataset_info in base_grid_config['datasets']:
        # Itera sui modelli *attualmente attivi* nel file di configurazione
        for model_name, specific_params in model_specific_search_space.items():
            model_base_config = fixed_params.copy()
            model_base_config.update({'dataset_name': dataset_info['name'], 'dataset_path': dataset_info['path'],
                                      'num_classes': dataset_info['num_classes'], 'model_name': model_name})
            current_search_space = {**common_search_space, **specific_params}

            for hyper_config in generate_configurations(model_base_config, current_search_space):
                total_generated_configs += 1
                if VERBOSE_DUPLICATE_CHECK: print("-" * 50,
                                                  f"\n[{total_generated_configs}] Generated Raw Config:\n{json.dumps(hyper_config, indent=2)}")

                if hyper_config.get('aggregation_algorithm') != "FedProx": hyper_config['fedprox_mu'] = 0.0
                normalize_partition_keys(hyper_config)
                normalize_fipa_keys(hyper_config)
                normalize_augmentation_keys(hyper_config)
                if 'ResNet' in model_name or 'GoogLeNet' in model_name or 'AlexNet' in model_name:
                    # setdefault, NOT assignment: this block only normalizes keys so
                    # that fingerprints match across runs. A hard assignment here
                    # overrode whatever image_size the search space asked for,
                    # collapsing the axis to 224 and making the generated configs
                    # duplicates that the fingerprint check then discarded.
                    # Mirrors the CSV side, which already uses setdefault.
                    hyper_config.setdefault('image_size', 224)
                    hyper_config.setdefault('convnet_hidden1', -1)
                    hyper_config.setdefault('convnet_hidden2', -1)
                elif 'ConvNet' in model_name:
                    hyper_config.setdefault('num_custom_layers', -1)
                if VERBOSE_DUPLICATE_CHECK: print(f" -> Normalized Config:\n{json.dumps(hyper_config, indent=2)}")

                fingerprint = get_config_fingerprint(hyper_config, all_possible_search_keys)

                if VERBOSE_DUPLICATE_CHECK: print(f" -> Fingerprint: {fingerprint}")

                if fingerprint in executed_fingerprints:
                    if VERBOSE_DUPLICATE_CHECK: print(" -> STATUS: DUPLICATE - Skipping.")
                    continue

                if VERBOSE_DUPLICATE_CHECK: print(" -> STATUS: NEW - Adding to queue.")
                executed_fingerprints.add(fingerprint)
                task_queue.put(hyper_config)
                configs_to_run_count += 1

    print("=" * 80)
    print(f"Generated {total_generated_configs} total configurations.")
    print(f"Skipped {total_generated_configs - configs_to_run_count} already executed or redundant configurations.")
    print(f"Adding {configs_to_run_count} new unique configurations to the execution queue.")
    print("=" * 80)

    # Count the puts, do not ask the queue. `multiprocessing.JoinableQueue.put`
    # hands the object to a background feeder thread that pickles it into a
    # pipe, so `empty()` reports the state of the *pipe*, not of the puts: right
    # after filling the queue it can still answer True and the whole grid exits
    # with "no new configurations" having just printed how many it added. The
    # race is timing dependent, which is why the same command runs one time and
    # exits the next.
    if configs_to_run_count == 0:
        print("=== No new configurations to run. Exiting. ===")
        return

    processes = []
    num_parallel = int(base_grid_config.get('num_parallel_executions',
                                            DEFAULT_NUM_PARALLEL_EXECUTIONS))
    print(f"\n=== Starting {num_parallel} Grid Search workers ===")
    base_port = base_grid_config['port']
    for i in range(num_parallel):
        process = multiprocessing.Process(target=run_grid_search_worker, args=(i, task_queue, base_port, csv_lock))
        processes.append(process);
        process.start()
    task_queue.join()
    for _ in range(num_parallel): task_queue.put(None)
    for process in processes: process.join()
    print("\n=== All parallel executions have finished. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main(sys.argv[1] if len(sys.argv) > 1 else GRID_SEARCH_CONFIG_PATH)