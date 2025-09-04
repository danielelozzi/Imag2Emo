# EEG Affective Computing Analysis Pipeline

## Overview

This project provides a comprehensive and configurable Python-based pipeline for EEG-based affective computing research. It handles the entire workflow from raw data processing to model training, hyperparameter optimization (HPO), and detailed reporting. The pipeline is designed to be modular and extensible, supporting multiple datasets (DEAP, GRAZ), various deep learning models, and flexible training scenarios.

## Features

-   **Multi-Dataset Support**: Includes preprocessing scripts for the DEAP (from both `.dat` and `.set` files) and GRAZ datasets.
-   **Flexible Data Preparation**: Configurable data segmentation, resampling, band-pass filtering, and channel selection.
-   **Advanced Training Scenarios**: Supports multiple cross-validation strategies:
-   **Multi-Class Classification**: Supports both binary classification (e.g., High/Low Valence) and 4-class classification based on the Valence-Arousal model (HVHA, HVLA, LVHA, LVLA).
    -   `k_simple`: A simple repeated train/validation/test split.
    -   `kfold`: Stratified K-Fold cross-validation.
    -   `loso`: Leave-One-Subject-Out cross-validation.
-   **Model Variety**: Easily integrates several common EEG deep learning models like EEGNetv4, Conformer, ContraNet, etc., through a simple wrapper.
-   **Hyperparameter Optimization**: Built-in HPO using Optuna to automatically find the best model parameters for each dataset.
-   **Comprehensive Reporting**: Generates learning curves, confusion matrices, classification reports, and summary statistics for each experiment, with TensorBoard integration for real-time monitoring.
-   **Modular Design**: The code is organized into logical modules for data loading, training, processing, and reporting, making it easy to understand and extend.

## Project Structure

The project is organized into several key Python scripts, each with a specific responsibility:

<details>
<summary><strong>File Descriptions</strong></summary>

-   `main.py`
    -   **Role**: The main entry point and orchestrator of the entire pipeline.
    -   **Functionality**: Controls the execution flow, from data preprocessing to HPO and final model training. It reads the main configuration, iterates through the defined experiments, and calls the appropriate modules.

-   `data_loader.py`
    -   **Role**: Handles data loading, splitting, and balancing.
    -   **Functionality**: Loads the pre-segmented data from disk, applies balancing strategies (e.g., custom undersampling to handle class imbalance), and splits the data into training, validation, and test sets according to the specified scenario (`kfold`, `loso`, `k_simple`). It also handles the binarization of labels into 2 or 4 classes.

-   `eeg_classifier_training.py`
    -   **Role**: Contains the core training and evaluation loop.
    -   **Functionality**: Manages model initialization, optimizer and loss function setup, the training/validation cycle, early stopping based on validation loss, and final evaluation on the test set. It also integrates with TensorBoard for logging.

-   `eegmodels.py`
    -   **Role**: A factory or wrapper for various EEG deep learning models.
    -   **Functionality**: Allows selecting a model by name from the configuration file and initializes it with the correct parameters (e.g., number of channels, classes, time points). This makes it easy to switch between different architectures like `EEGNetv4`, `Conformer`, `ContraNet`, etc.

-   `prepare_training_segments.py`
    -   **Role**: Prepares final training segments (windows) from preprocessed data.
    -   **Functionality**: Takes the subject-level `.npy` files and applies global preprocessing steps like resampling and filtering. It then slices the data into smaller, fixed-length windows suitable for model training and saves them to a cache directory for fast access.

-   `processa_deap.py` & `processa_deap_bdf.py`
    -   **Role**: Initial data converters for the DEAP dataset.
    -   **Functionality**: These scripts handle the initial conversion of the raw DEAP dataset files (`.set` or `.dat` pickle files) into a standardized `.npy` format. They extract EEG data and corresponding labels (both public and private ratings).

-   `processa_graz.py`
    -   **Role**: Initial data converter for the GRAZ dataset.
    -   **Functionality**: This script converts the raw GRAZ dataset files (`.set` and `.csv` labels) into the standardized `.npy` format, including logic to exclude specific trials based on a list of similar images.

-   `reporting.py`
    -   **Role**: A utility module for generating all visual and text-based reports.
    -   **Functionality**: Creates and saves plots for learning curves (loss and accuracy), class distributions, and confusion matrices. It also saves detailed classification reports and summary metrics (mean, std) to CSV files.

</details>

## Workflow

The pipeline is orchestrated by `main.py` and follows these steps:

1.  **Configuration**: The user sets up global parameters, file paths, and pipeline control switches (e.g., `RUN_DISK_SEGMENTATION`, `RUN_HPO`) directly in `main.py`. This includes defining the datasets, models, and scenarios to run.

2.  **Initial Data Processing (Optional)**: If `RUN_DISK_SEGMENTATION` is `True`, the `processa_*.py` scripts are executed. They convert the raw dataset files (DEAP, GRAZ) into an intermediate `.npy` format, where each file typically represents one trial for one subject. This step only needs to be run once.

3.  **Segment Generation (Optional)**: Next, `prepare_training_segments.py` is called. It loads the intermediate `.npy` files, applies global preprocessing like resampling or filtering, and slices the data into the final training windows (e.g., 2-second segments). These segments are saved to a disk cache to speed up subsequent runs. This step also only needs to be run once.

4.  **Hyperparameter Optimization (Optional)**: If `RUN_HPO` is `True`, the pipeline initiates an Optuna study for each dataset. It uses a subset of the data and a simplified training scenario (`k_simple`) to efficiently find the best hyperparameters (e.g., learning rate, dropout). The best parameters found are saved to a `hpo_best_params.json` file. If `RUN_HPO` is `False`, the pipeline attempts to load these parameters from the JSON file instead.

5.  **Full Training & Evaluation**: This is the main experimental phase. The pipeline iterates through all configured combinations (e.g., `[GRAZ]-[PRIVATE]-[valence]-[loso]-[EEGNetv4]`).
    -   **2-Class**: `[GRAZ]-[PRIVATE]-[valence]-[loso]-[EEGNetv4]`
    -   **4-Class**: `[DEAP_BDF]-[PUBLIC]-[valence_arousal_4class]-[kfold]-[EEGNetv4]`
    -   For each combination, it uses `get_data_splits` from `data_loader.py` to yield the train/validation/test sets for the current fold or run.
    -   It then calls `train_and_evaluate_model`, which performs the training using the best hyperparameters found during HPO (or defaults).
    -   The training loop includes features like early stopping to prevent overfitting and a learning rate scheduler.

6.  **Reporting**: After each fold is trained, the `reporting.py` module is used to save plots (learning curves, confusion matrices) and metrics for that specific fold. Once all folds for a given scenario are complete, it generates aggregate reports, including average performance metrics across folds and averaged learning curves with standard deviation.

## Configuration

All major configurations are centralized in `main.py`. Key dictionaries to modify include:

-   **Control Switches**:
    -   `RUN_DISK_SEGMENTATION`: Set to `False` to skip the initial data processing and segmentation if the data is already cached on disk.
    -   `RUN_HPO`: Set to `False` to skip hyperparameter optimization and use pre-optimized parameters from a JSON file (or defaults).

-   **`PIPELINE_CONFIG`**: Defines parameters for the overall pipeline. Key settings include:
    -   `model_types`: A list of models to test (e.g., `['EEGNetv4']`).
    -   `binarized_threshold`: Controls the classification task.
        -   For **2-class** classification, use a float (e.g., `5.0`).
        -   For **4-class** classification, use a dictionary with thresholds for both dimensions (e.g., `{'valence': 5.0, 'arousal': 5.0}`).
    -   `balancing_strategy`: Strategy for handling imbalanced classes (e.g., `'custom_undersampling'`).
    -   Resampling/filter settings and HPO parameters.

-   **`TRAINING_CONFIG`**: Contains parameters for the training loop itself, such as the number of epochs, batch size, learning rate, and optimizer settings.

-   **`datasets_to_train`**: A dictionary specifying which datasets to run experiments on. For each dataset, you define:
    -   Paths to the segmented data.
    -   `public_labels` and `private_labels`: A list of label metrics to be tested. To run a 4-class experiment, include `'valence_arousal_4class'` in this list.
    -   Other specific metadata like `sampling_rate`.

-   **`*_TRAINING_PREP_CONFIG`**: Dictionaries that define the segmentation parameters for each dataset (e.g., start/end times, segment length).

### Example: Setting up a 4-Class Experiment

To run a 4-class classification on the DEAP_BDF dataset with public labels:

1.  In `main.py`, ensure the `PIPELINE_CONFIG` is set for 4-class classification:
    ```python
    PIPELINE_CONFIG = {
        # ... other settings
        'binarized_threshold': {'valence': 5.0, 'arousal': 5.0},
        # ...
    }
    ```

2.  In the `datasets_to_train` dictionary, make sure `'valence_arousal_4class'` is included in the `public_labels` list for `DEAP_BDF`:
    ```python
    datasets_to_train = {
        'DEAP_BDF': {
            # ... other settings
            'public_labels': ['valence_arousal_4class'],
            'private_labels': [], # Can be empty if not testing private labels
            # ...
        },
        # ... other datasets
    }
    ```

The pipeline will automatically detect this configuration, load both valence and arousal labels, combine them into 4 classes (HVHA, HVLA, LVHA, LVLA), and train the model accordingly.

## How to Run

1.  Ensure all dependencies are installed.
2.  Update the paths to the raw datasets and other configuration variables in `main.py` as needed.
3.  Execute the main script from your terminal:

    ```bash
    python main.py
    ```

All results, reports, and logs will be saved in the `TRAINING_RESULTS` directory, organized by experiment configuration.

## Dependencies

This project relies on several key libraries. You can install them via `pip`:

```bash
pip install torch numpy pandas scikit-learn mne optuna matplotlib seaborn tensorboard
```

-   **PyTorch**: For building and training the neural networks.
-   **Optuna**: For hyperparameter optimization.
-   **MNE-Python**: For reading EEG data files (e.g., `.set` format).
-   **Scikit-learn**: For data splitting, metrics, and scaling.
-   **Pandas & NumPy**: For data manipulation.
-   **Matplotlib & Seaborn**: For plotting and generating reports.
-   **TensorBoard**: For logging and visualizing training progress.