# Terminology extraction project for Terminology and Ontology class 2021

![architecture](https://github.com/sarmilaupadhyaya/Terminology-Extraction/blob/main/images/training.png)

![architecture testing](https://github.com/sarmilaupadhyaya/Terminology-Extraction/blob/main/images/testing.png)

some from intro

# Outline

1. [Directory Structure](#directory-structure)
2. [Introduction](#introduction)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Contributing](#contributing)
6. [Licence](#licence)


## Directory structure

    .
    ```
    
    .
├── bilm_inference.py
├── bilm_train.py
├── data
│   ├── gold_annotation.csv
│   ├── silver_annotated
│   │   ├── test_tfidf.tsv
│   │   ├── test.tsv
│   │   ├── train_tfidf.tsv
│   │   └── train.tsv
├── dataloader.py
├── documents.py
├── extraction_terminology.py
├── gold_annotated.py
├── images
│   ├── testing.png
│   └── training.png
├── inference
│   ├── clean_inference.txt
│   ├── clean.py
│   ├── g.py
│   └── inference.txt
├── models
│   ├── bert-crf.py
│   ├── bilstm_crf.py
│   └── rule_based_extraction.py
├── output
│   ├── chkpt
│   │   ├── markable-bi-lstm-td-model-2.hdf5
│   │   ├── markable-bi-lstm-td-model.hdf5
│   │   ├── markable-bi-lstm-td-model-pointwise.hdf5
│   │   └── markable-bi-lstm-td-model-tfidf.hdf5
│   ├── extracted_gold.txt
│   ├── extracted_nn.txt
│   ├── extracted_rule.txt
│   ├── extracted_terminology_multiword_analysed.csv
│   ├── extracted_terminology_multiword.csv
│   ├── extracted_terms_multi_pointwise_analysed.csv
│   ├── extracted_terms_multi_tfidf_analysed.csv
│   ├── extracted_terms_multi_tfidf.csv
│   ├── incorrect_sentence_new.txt
│   ├── incorrect_sentence_tfidf.txt
│   ├── tag2idx.pk
│   └── tag2idx_tdidf.pk
├── params.py
├── README.md
├── rule_based_extraction.py
├── run_train.sh
├── silver_annotated.py
├── term_extraction.py
├── test.tsv
├── train.tsv
└── utils.py

8 directories, 78 files
    
    ```


## Introduction

    
![architecture](https://github.com/IuliiaZaitova/code_slingers/blob/master/images/New_Architecture_updated.png?raw=true)


## Installation

Note: Since, the project has three models as the pipeline running serially, it requires some memory space and RAM. Make sure you have around 3 GB physical disk space, 4 GB RAM and enough space to install the requirements. 

In order to get the model to run, follow these installation instructions.


<!-- ### Requirements -->
Pre-requisites:

    python>=3.7.5

On Windows you will also need [Git for Windows](https://gitforwindows.org/).

---
_Optional_: to install a specific version of Python:

#### Ubuntu:

    pyenv install 3.7.5

(To install ```pyenv```, follow [this tutorial](https://github.com/pyenv/pyenv-installer#installation--update--uninstallation), and then [this one](https://www.laac.dev/blog/setting-up-modern-python-development-environment-ubuntu-20/))
<!--     sudo apt-install python3.7 -->


#### Mac:

    brew install python@3.7


#### Windows:
Download Python 3.7.5 for Windows [here](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe), run it and follow the instructions.
    
---
### 1. Clone the repository

    git clone [link]

_Optional_: use the package manager [pip](https://pip.pypa.io/en/stable/) to install a vitual environment.

    bash
    pip install virtualenv
    
    
    
#### 2. Navigate to the folder with the cloned git repository

#### 3. Create Virtual Environment

    virtualenv <name of env> --python /usr/bin/python[version] or <path to your python if its not the mentioned one>
    
Conda:

    conda create --name <name of your env> python=3.7

#### 4. Activate Virtual Environment

    source name_of_env/bin/activate
On Windows:

    name_of_env\Scripts\activate
Conda:

    conda activate <name of your env>

(To leave the virtual environment, simply run: ```deactivate``` (with virtualenv) or ```conda deactivate``` (with Conda))

---

### 5. Install Requirements

    pip install -r requirements.txt
        
Conda:

    conda install pip
    pip install -r requirements.txt


---

_Optional_: If you're on Mac.  

***Note: if you are on mac, then first install wget using brew***  

    brew install wget

---
#### 6. You also need to download the spacy model:

    python -m spacy download en_core_web_lg

---

### 7. Initial Downloads

This is the step that downloades the checkpoint for the Neural Network model
    
    chmod +x initialization_script.sh
    ./initialization_script.sh
    
 On Windows:
 
 If you did not choose to import Linux tools when installing Git for Windows, you have to use the Git Bash to run the initialization script. Either way, both Git Bash and Windows Command Prompt should be run as administrator.

    chmod +x initialization_script.sh
    sh initialization_script_windows.sh


************************************************************************************************************************************
**NOTE**: If you encounter the following error (mostly due to internet issue models are not downloaded properly and you can't unzip)


    Archive:  models.zip
      End-of-central-directory signature not found.  Either this file is not
      a zipfile, or it constitutes one disk of a multi-part archive.  In the
      latter case the central directory and zipfile comment will be found on
      the last disk(s) of this archive.
    unzip:  cannot find zipfile directory in one of models.zip or
            models.zip.zip, and cannot find models.zip.ZIP, period.


Then Kindly rerun the following code


    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-bnWPP6-c42wVQMztHpcVquUGd1XGc-z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-bnWPP6-c42wVQMztHpcVquUGd1XGc-z"  -O models.zip && rm -rf /tmp/cookies.txt
    unzip models.zip


If not, then skip this section!
************************************************************************************************************************************
**_YAY!!_** Installation is done! Now you can jump to the execution part and run the web app.


## Execution

To run the inference on the gold dataset that is inside data/gold_annotation.csv, run the following code, being in the root directory.

    python3 

---


## Dataset

Dataset, all the articles used for training and rule based is inside data folder. The gold test article is inside inference/cleaned_inference.txt


<!-- - Evaluation of dataset -->



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)

