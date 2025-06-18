# KDL Classifier

Interactive System for Classifying German Clinical Documents based on the KDL Code System

## Introduction
Reliable classification of clinical text documents is crucial as it can greatly benefit medical staff in categorizing and retrieving documents based on previous records. In this paper, we propose an interactive system capable for classifying German clinical texts based on the hierarchical \textit{Klinische Dokumentenklassen-Liste} (KDL) labelling scheme. KDL is a structured list of clinical document types used in the healthcare sector, for example, in the context of patient treatment in hospitals and clinics. Different individual names for medical documentation and records are used in practice, and the clinical document class list serves as a standardization tool with the goal of uniformly classifying data.


## Installation


#### Used Python Version
Python 3.12.2

#### Requirements

To install all required packages run:

``` bash
pip install -r requirements.txt
```

#### Creating a dedicted Python environemnt (recommended)
Create Environment
``` bash
python3.12 -m venv "kdl_tool"
```
Activate Environment
``` bash
source kdl_tool/bin/activate
```
Upgrade Pip
``` bash
pip install --upgrade pip
```


## How to run
``` bash
streamlit run KDLClassifcationApp.py
```


Install requirements
- Our Expermients were ran using Pytorch (CUDA 12.6)

``` bash
pip install -r requirements.txt
```
