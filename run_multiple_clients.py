import os
import time
import threading
import json
from typing import Dict, List

from data_splitter import DatasetSplitter
from federated_client import FederatedClient

def load_json(filename: str) -> Dict:
    """Loads a JSON file."""
    with open(filename) as f:
        return json.load(f)


def start_client_thread(config: Dict, dataset_path: str) -> None:
    """
    Initializes and runs a FederatedClient in a separate thread.
    """
    print(f"Starting client for dataset: {dataset_path}")
    try:
        FederatedClient(config, dataset_path)
    except Exception as e:
        print(f"Error starting client for {dataset_path}: {e}")
        import traceback
        traceback.print_exc()


def main(config: Dict) -> None:
    """
    Main function to split the dataset and launch multiple client simulations.
    """
    num_clients = config['num_clients']
    splitting_dir = config['splitting_dir']
    source_images_dir = config['dataset_path'] # <-- Modifica chiave

    print(f"Preparing to split the dataset from '{source_images_dir}' for {num_clients} clients into '{splitting_dir}'.")
    splitter = DatasetSplitter(
        output_base_dir=splitting_dir,
        source_images_dir=source_images_dir,
        num_clients=num_clients
    )
    splitter.split_dataset()
    print("Dataset splitting complete.")

    threads: List[threading.Thread] = []
    for i in range(num_clients):
        client_dataset_path = os.path.join(splitting_dir, f'client_{i}')
        thread = threading.Thread(
            target=start_client_thread,
            args=(config, client_dataset_path)
        )
        thread.start()
        threads.append(thread)
        time.sleep(2)

    for thread in threads:
        thread.join()

    print("All client threads have finished.")


if __name__ == '__main__':
    print("This script is intended to be run by 'federated_grid_search.py'.")