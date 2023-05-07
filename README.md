# CS598_Project

Replicating _"Design and implementation of a deep recurrent model for prediction of readmission in urgent care using electronic health records"_ (Zebin and Chaussalet, 2019)

## Model Training and Evaluation

**Note: To run this project, you will need a copy of the MIMIC III dataset located
on the machine you wish to run. The MIMIC III dataset is publicly available through
credentialed access on Physionet.org.**

### 0. Installing Dependencies

This project requires:

| Dependency   | Version          |
|--------------|------------------|
| python       | 3.11 _or higher_ |
| matplotlib   | 3.7.1            |
| numpy        | 1.24 _or higher_ |
| p_tqdm       | 1.4.0            |
| pandas       | 2.0 _or higher_  |
| scikit-learn | 1.2.2            |
| torch        | 2.0.0            |
| torchvision  | 2.0.0            |
| openpyxl     | 3.1.2            |

Additionally, you will need to install [Jupyter](https://github.com/jupyter/jupyter)
to run the interactive notebooks.

### 1. Subsampling the Data

If you already have a suitably sized subset of the _MIMIC III_ dataset, you can skip this step.

By running the `data_sampling/datasampler.py` script, you can generate a subset of the full _MIMIC III_ dataset
that will more easily run on your machine. We used a sample size of `10000` ICU stays.

**Note: you will need to edit this script with the correct path to the copy of the MIMIC III
dataset that you downloaded on your machine.**

Running the data sampler:
```bash
python3 data_sampling/datasampler.py
```

### 2. Preprocessing the Data

The data must be preprocessed and stored into binary files before it is used by
the models. The following scripts from the `data_cleaning` directory should be run
before launching the interactive notebooks:

```bash
python3 data_cleaning/preprocessing_mp.py
python3 data_cleaning/events_to_list.py
```

### 3. Running the Interactive Python Notebooks

The training of the various models is performed in Jupyter interactive notebooks.
To run one of these models, launch the Jupyter server with file of the model you
wish to train:

```bash
# Open LSTM+CNN Model interactive notebook
jupyter notebook source/lstm-cnn-model.ipynb

# Open LSTM Model baseline interactive notebook
jupyter notebook source/lstm-model.ipynb

# Open Logistic Model baseline interactive notebook
jupyter notebook source/logistic-model.ipynb
```

The results history of the models are stored as pickle binary files in the
same directory.

## Repository Organization

This project repository is organized into the following sub-folders:

* __data__: binary working folders of preprocessed data
* __demo_data__: binary working folders of preprocessed data from publicly available demo data used for model development
* __data_sampling__: data subset sampling source code
* __data_cleaning__: data preprocessing source code
* __project_summary__: tables and figures summarizing reproducibility results
* __source__: interactive notebooks for each model and utility classes

## Citation

T. Zebin and T. J. Chaussalet, "Design and implementation of a deep recurrent model for prediction of readmission in urgent care using electronic health records," 2019 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Siena, Italy, 2019, pp. 1-5, doi: 10.1109/CIBCB.2019.8791466.
