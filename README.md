# Master Arbeit

Firat Metin - Quality Assessment Sensor Fusion for Unobstrusive Cardiorespiratory Signals

# Task Description:

- Implementation of motion artifact detection / signal assesment algorithms
- Development of a signal fusion algorithm for motion artifact compensation
- Evaluation of the methods including labelling of the data for reference

# Workflow:
Data ---> Quality Assesment ---> MA Detection ---> MT Gaussian Process ---> HR & RR

# To set the virtual environment up:
Creating and activating the virtual environment:

``` (Terminal)
1. python -m venv venv
2. . .\venv\Scripts\activate.bat
```

Installing required packages:

1. This project developed by using python version 3.11.9.

2. Please install ipykernel package, if you want to work with the jupyter notebooks:
    1.1: Simply running a cell in the jupyter notebook will show you the installation prompt.

3. For the required modules, please check `requirements.txt` and pip install them individually.

4. If the image-saving property in the labeler gui wanted to be used, a TeX system (preferably MiKText in Windows) must be installed on the computer.