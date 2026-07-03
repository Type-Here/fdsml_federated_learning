import os
import pandas as pd
import shutil
import re
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Dict


class DatasetSplitter:
    """
    Splits an image dataset into multiple client-specific subsets for federated learning.

    This class now automatically detects classes from the source directory structure,
    builds an internal annotation mapping, performs a stratified split, and copies
    the corresponding image files into a directory structure suitable for training.
    """

    def __init__(self, output_base_dir: str, source_images_dir: str, num_clients: int):
        """
        Args:
            output_base_dir (str): The root directory where client data folders will be created.
            source_images_dir (str): The directory containing the source images, with subdirectories for each class.
            num_clients (int): The number of clients to split the data for.
        """
        self.output_base_dir = output_base_dir
        self.source_images_dir = source_images_dir
        self.num_clients = num_clients
        self.class_map: Dict[str, int] = {}
        self.index_to_dir_map: Dict[int, str] = {}

        os.makedirs(self.output_base_dir, exist_ok=True)

        # La logica del CSV è stata sostituita da questa funzione
        self.dataframe = self._build_dataframe_from_folders()
        if self.dataframe.empty:
            raise ValueError(f"No images found in source directory: {self.source_images_dir}")

    def _build_dataframe_from_folders(self) -> pd.DataFrame:
        """
        Scans the source directory to build a dataframe of filenames and class labels.
        It determines class indices based on directory names (numeric first, then alphabetical).
        """
        print(f"Scanning '{self.source_images_dir}' to infer classes from directories...")

        # 1. Trova le sottocartelle che rappresentano le classi
        try:
            class_dirs = [d for d in os.listdir(self.source_images_dir)
                          if os.path.isdir(os.path.join(self.source_images_dir, d))]
        except FileNotFoundError:
            print(f"Error: Source image directory not found at {self.source_images_dir}")
            raise

        if not class_dirs:
            print(f"Error: No class subdirectories found in {self.source_images_dir}")
            return pd.DataFrame()

        # 2. Tenta di mappare le classi in base a un numero nel nome della cartella
        numeric_map = {}
        can_use_numeric_map = True
        for dir_name in class_dirs:
            match = re.search(r'\d+', dir_name)
            if match:
                numeric_map[dir_name] = int(match.group(0))
            else:
                can_use_numeric_map = False
                break

        # 3. Se il mappaggio numerico non è possibile, usa l'ordine alfabetico
        if can_use_numeric_map and len(set(numeric_map.values())) == len(class_dirs):
            print("Using numeric values found in directory names for class mapping.")
            self.class_map = numeric_map
        else:
            print("Could not infer classes from numbers in directory names. Falling back to alphabetical order.")
            sorted_dirs = sorted(class_dirs)
            self.class_map = {dir_name: i for i, dir_name in enumerate(sorted_dirs)}

        print("--- Class Mapping Detected ---")
        for dir_name, class_idx in self.class_map.items():
            print(f"  '{dir_name}' -> Class {class_idx}")
        print("----------------------------")

        self.index_to_dir_map = {v: k for k, v in self.class_map.items()}

        # 4. Popola il DataFrame con filename e classe
        records = []
        for dir_name, class_idx in self.class_map.items():
            class_path = os.path.join(self.source_images_dir, dir_name)
            for filename in os.listdir(class_path):
                # Assicurati di includere solo file di immagine comuni
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    records.append({'filename': filename, 'class': class_idx})

        return pd.DataFrame(records)

    def _copy_images(self, indices: List[int], client_dir: str, split_name: str) -> None:
        """
        Copies image files to the appropriate train/validation directory for a client.
        This version is more robust and uses the original class directory names.
        """
        for index in indices:
            row = self.dataframe.loc[index]
            filename = row['filename']
            class_index = row['class']

            # Ottieni il nome della cartella originale dall'indice della classe
            class_dir_name = self.index_to_dir_map.get(class_index)
            if not class_dir_name:
                print(f"Warning: Could not find directory name for class index {class_index}. Skipping.")
                continue

            destination_dir = os.path.join(client_dir, split_name, class_dir_name)
            os.makedirs(destination_dir, exist_ok=True)

            source_image_path = os.path.join(self.source_images_dir, class_dir_name, filename)
            destination_image_path = os.path.join(destination_dir, filename)

            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
            else:
                print(f"Warning: Source image not found and will be skipped: {source_image_path}")

    def _clear_existing_split(self) -> None:
        """
        Deletes all contents of the output directory to ensure a clean split.
        Warning: This is a destructive operation.
        """
        print(f"Clearing existing data in '{self.output_base_dir}'...")
        if os.path.exists(self.output_base_dir):
            for item_name in os.listdir(self.output_base_dir):
                item_path = os.path.join(self.output_base_dir, item_name)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  - Deleted directory: {item_path}")
        print("Clearing complete.")

    def split_dataset(self, validation_split_size: float = 0.1) -> None:
        """
        Performs the main dataset splitting operation.

        It first clears any previous splits, then uses StratifiedKFold to divide
        the data among clients, and for each client's data, performs a further
        split into training and validation sets.
        """
        self._clear_existing_split()

        y = self.dataframe['class']
        min_class_count = y.value_counts().min()
        if min_class_count < self.num_clients:
            print(f"Warning: Least populated class has {min_class_count} samples, "
                  f"less than num_clients={self.num_clients}. "
                  f"Reducing effective clients to {min_class_count} for splitting.")
            effective_splits = min_class_count
        else:
            effective_splits = self.num_clients

        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)

        # L'indice del DataFrame viene usato per lo split
        X = self.dataframe.index

        for i, (_, client_indices) in enumerate(skf.split(X, y)):
            client_dir = os.path.join(self.output_base_dir, f"client_{i}")
            os.makedirs(client_dir, exist_ok=True)
            print(f"Processing data for client_{i}...")

            client_labels = y.loc[self.dataframe.index[client_indices]]
            min_class_in_client = client_labels.value_counts().min()
            can_stratify = min_class_in_client >= 2 and len(client_indices) >= 10

            try:
                train_indices, valid_indices = train_test_split(
                    client_indices,
                    test_size=validation_split_size,
                    random_state=42,
                    stratify=client_labels if can_stratify else None
                )
            except ValueError as e:
                print(f"  Warning: train_test_split fallback for client_{i}: {e}")
                train_indices = client_indices[:-1]
                valid_indices = client_indices[-1:]

            # Usiamo gli indici del DataFrame per copiare i file
            self._copy_images(self.dataframe.index[train_indices], client_dir, "train")
            self._copy_images(self.dataframe.index[valid_indices], client_dir, "valid")
            print(f"  - Created train set with {len(train_indices)} samples.")
            print(f"  - Created valid set with {len(valid_indices)} samples.")