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

```
	.
	├── bilm_inference.py  #inference for pur Neural Network model
	├── bilm_train.py     # training bilm model
	├── data              # folder that contain all artikles and gold annotation
	│   ├── gold_annotation.csv   # gold annotation with tags
	│   ├── silver_annotated     # folder with silver annotated data
	│   │   ├── test_tfidf.tsv   # tfidf train
	│   │   ├── test.tsv         # frequency test
	│   │   ├── train_tfidf.tsv  # tfidf test
	│   │   └── train.tsv        # frequency train
	├── dataloader.py            # dataloader got neural net model
	├── documents.py             # class for documents
	├── extraction_terminology.py # extracting candidate and filtering
	├── gold_annotated.py         # 
	├── images                   # architecture
	│   ├── testing.png
	│   └── training.png
	├── inference             # raw inference text
	│   ├── clean_inference.txt # clean inference
	│   ├── clean.py
	│   ├── g.py
	│   └── inference.txt
	├── models   # model for Neural Net
	│   ├── bilstm_crf.py
	├── output # contains checkpoint and extracted terms from different method
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
	├── params.py # all parameters
	├── README.md
	├── rule_based_extraction.py  # rule based extraction
	├── run_train.sh  # script to train NN model
	├── silver_annotated.py  # script to extract silver annotated train test data
	├── term_extraction.py  # main extraction file for gold annotation
	└── utils.py # contains function to get iob from terms and terms from iob data

    
 ```


## Introduction

    
Many fields employ specialized vocabulary that are either not recognized outside of a given community of specialists, or have a somewhat different sense. The elements of these specialized vocabulary are known as terms. Term extraction is the task of identifying terms in a given document or corpus. This task can be performed in either a monolingual or multilingual setting, and can be a first step toward other tasks, such as topic modelling, automatic summarization, ontology construction, and machine translation.

## Installation


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

#### 7. Then, download the checkpoints from this link ![Link](https://drive.google.com/file/d/1W8g7d60hu8I9txYtfrcqLqUKTbTz02SA/view?usp=sharing)
extract and put the folder inside output/
---


**_YAY!!_** Installation is done! Now you can jump to the execution part and run the web app.


## Execution
 ** Note ** initial running may take a while as it has to download elmo model which takes a while.

To run the inference on the gold dataset that is inside data/gold_annotation.csv, run the following code, being in the root directory.

    python3 term_extraction.py --filter_method <tfidf or frequency> --gold_annotation_path <data/gold_annotation.csv>


---


## Dataset

Dataset, all the articles used for training and rule based is inside data folder. The gold test article is inside inference/cleaned_inference.txt
further read data instructions file to know more: ![Link](https://github.com/sarmilaupadhyaya/Terminology-Extraction/blob/main/data_instructions.txt)



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)

