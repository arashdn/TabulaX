# TabulaX: Leveraging Large Language Models for Multi-Class Table Transformations

This repository contains resources developed within the following paper:

    A. Dargahi Nobari, and D. Rafiei. "TabulaX: Leveraging Large Language Models for Multi-Class Table Transformations", VLDB, 2025
	
You may check the [paper](#) ([PDF](https://arxiv.org/abs/2411.17110)) for more information.


## Requirements

Several libraries are used in the project. You can use the provided `environment.yml` file to create the conda environment for the project or use `requirements.txt` to install the dependencies with pip. 
If you prefer not to use the environment file, the environment can be set up by the following command.
```
conda create -n tgen python=3.11
conda activate tgen
conda install pip
pip install nltk==3.8.1
pip install numpy==1.26.4
pip install openai==1.35.7
pip install scikit-learn==1.5.1
pip install scipy==1.13.1
pip install tqdm==4.66.4 # if you will use tqdm
pip install numpy==1.26.4
pip install pandas==2.2.3

# requirements.txt created by: pip list --format=freeze > requirements.txt
# environment.yml created by: conda env export | grep -v "^prefix: " > environment.yml
```


## Usage

Two main directories are in the repo: `data`, and `src`.


### data
All datasets are included in this repo in `Datasets` directory. Each dataset contains several tables (each as a folder) and each table contains `source.csv`, `target.csv`, and `ground truth.csv`. The datasets available in this file are:
- `AutoJoin`: The Web Tables (WT) dataset.
- `FlashFill`: The Spreadsheet (SS) dataset.
- `ALL_TDE`: The Table Transformation (TT) dataset.
- `DataXFormer`: The Knowledge Based Web Tables (KBWT) dataset.

### src
The source files are located in `src/LLM_pipeline` directory.
If you are using GPT models, make sure your OpenAI API key is stored inside `openai.key` file inside this directory.
If you are using LLaMA 3 models, You need to use vLLM to serve the model on local.

If you need to run the code, you just need to use `run_pipeline.py` (see the Running the pipeline section). To run the input classifier use the `*_classifier.py` scripts in classifier directory.

This directory contains three sub-directories.


##### classifier
The resources required for automatic input table classification.

* `prompts/*`: Prompt templates for each classification model.
* `classifierutil.py`: The library for general function required for classification.
* `DFX_classes.csv`: Ground truth classes for KBWT dataset.
* `TDE_classes.csv`: Ground truth classes for TT dataset.
* `report_metrics.py`: Helper functions to generate the classification performance at the end of classification are in this library.

* `gpt_classifier.py`: Run this file to use GPT models for input classification. Some values may be edited inside the file.
  * `MODEL_NAME`: The name and version for the GPT model. Default is `"gpt-4o-2024-05-13"`.
  * `PROMPT_CACHE_PATH`: The cache directory for model prompts. Make sure the directory exists. The default value is `BASE_PATH / "cache/classifier_prompts"`
  * `ALL_CLASSES_JSON`: The path for the output file with predicted classes.
  * `EXAMPLE_SIZE`: Number of examples provided to facilitate the classification. The default value is 5.
  * `DS_PATHS`: A list of the path to datasets to be classified.
  
* `llama_classifier.py`: Run this file to use LLaMA 3 models for input classification. Some values may be edited inside the file.
  * `MODEL_NAME`: The name and version for the LLaMA model. Default is `"llama3.1-8b"`.
  * `PROMPT_CACHE_PATH`: The cache directory for model prompts. Make sure the directory exists. The default value is `BASE_PATH / "cache/classifier_prompts"`
  * `ALL_CLASSES_JSON`: The path for the output file with predicted classes.
  * `EXAMPLE_SIZE`: Number of examples provided to facilitate the classification. The default value is 5.
  * `DS_PATHS`: A list of the path to datasets to be classified.


##### transformers
The libraries including transformation functions. Functions and modules in this directory will be imported in the other script and are not executables.

* `prompts/*`: Prompt templates for each LLM given each transformation class.
* `basic.py`: The functions and components to generate output by directly prompting an LLM.
* `algorithmic.py`: The functions and components to generate transformations for algorithmic inputs.
* `general.py`: The functions and components to generate output for general-class inputs.
* `numeric.py`: The functions and components to generate transformations for numeric inputs.
* `string.py`: The functions and components to generate transformations for string inputs.



##### util
The misc helper libraries. Functions and modules in this directory will be imported in the other script and are not executables.

* `dataset.py`: The functions and components to load and sample the data.
* `JoinEval.py`: The functions to evaluate and report metrics on table join.



### Running The pipeline
To run the transformation pipeline, run the `run_pipeline.py` script.
Some values may be edited inside the file.
* `ED_CACHE_PATH`: The cache file path for edit distance values. Make sure the directory exists. The default value is `BASE_PATH / "cache/edit_distance/ed.pkl"`
* `MODEL_NAME`: The LLM that is used for transformation generation. Supported models are `"gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "llama3.1-8b"`.
* `BASIC_PROMPT`: If the value is set to true, the code will basically prompt the LLM instead of running the framework. This is only used for the baselines and should be set to `False`
* `EXAMPLE_SIZE`: Number of examples provided to facilitate the transformation. The default value is 5.
* `MATCHING_TYPE`: The matching strategy for joining tables. Supported values are `["edit_dist", "exact"]`.
* `CLASSIFICATION_TYPE`: The classification approach for joining tables. Supported values are `['golden', gpt_classifier']`. "golden" uses ground truth classifier.
* `DS_PATH`: Path for the dataset directory.
* `OUTPUT_DIR`: The path to store output files and performance report.




## Citation

Please cite the paper, If you used the codes in this repository.

