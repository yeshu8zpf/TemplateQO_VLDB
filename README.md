# TemplateQO: Template-aware Query Optimization

## Overview

TemplateQO is a machine learning-driven query optimization framework that categorizes query plans into templates and uses a unified set of candidate plans for each template. By avoiding the complexity of encoding query plan trees, it simplifies the sample space and allows for faster performance improvements with more training data.

## Setup

To set up the development environment for TemplateQO, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/sigmod2026/TemplateQO.git
cd TemplateQO.git
```

### 2. Create the conda environment

To create the conda virtual environment with the required dependencies, use the following command:

```bash
conda env create -f environment.yml
```

This will create the environment and install the necessary packages.

### 3. Activate the environment

After the environment has been created, activate it with:

```bash
conda activate <your_env_name>
```

### 4. Install `pilotscope`

After creating the environment, you will need to manually install `pilotscope`. For installation instructions, refer to the following link:

[Install Pilotscope](https://github.com/alibaba/pilotscope?tab=readme-ov-file#installation)

Follow the provided steps on the GitHub page to complete the installation.

## Re-produce the Project

To reproduce the results with the IMDb database, run the following command:

```bash
python main.py --database imdb
```

## Use Your Own Dataset

If you want to use your own dataset, follow these steps:

### 1. Generate Candidate Hints

Run the `generate_hint.py` script to generate hints set for a template from your randomly generated SQLs:

```bash
python generate_hint.py \
    --sql_path <your_sql_file_path> \
    --hint_path <your_hint_output_path> \
    --alias_map_path <your_alias_map_path> \
    --database <database_name>
```

This will generate the hints required for generating candidate plan sets.

### 2. Generate the Candidate Plan Set

Run the `generate_plan.py` script to generate the necessary files ( `plan.txt`, `time.txt`) from your dataset:

```bash
python generate_plan.py \
    --sql_path <your_sql_file_path> \
    --hint_path <your_hint_file_path> \
    --plan_path <your_plan_output_path> \
    --time_path <your_time_output_path> \
    --database <database_name> \
    --timeout <timeout_duration>
```

This will generate the time.txt and plan.txt required for training and evaluation.

### 3. Organize Your Dataset

After generating the files, organize them into the following directory structure:

- `<your_train>` directory: Contains one folder per template, with each folder containing the three files (`sql.txt`, `plan.txt`, `time.txt`) for that template.
- `<your_test>` directory: Contains a similar structure as the training set, with folders for each template and the corresponding files.

Example directory structure:

```
<your_train>/
    ├── template_1/
    │   ├── sql.txt
    │   ├── plan.txt
    │   ├── time.txt
    ├── template_2/
    │   ├── sql.txt
    │   ├── plan.txt
    │   ├── time.txt
<your_test>/
    ├── template_1/
    │   ├── sql.txt
    │   ├── plan.txt
    │   ├── time.txt
    ├── template_2/
    │   ├── sql.txt
    │   ├── plan.txt
    │   ├── time.txt
```

### 4. Modify `remain_filter_columns`

The `remain_filter_columns` file contains a dictionary where the keys are the table names, and the values are lists of column names in each table. You need to update this dictionary to include the columns from your dataset that are involved in filtering.

