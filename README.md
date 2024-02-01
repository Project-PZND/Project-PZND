# Machine Learning in Picture Recognition

by Jakub Węgrzynek, Maria Pardo, Natalia Szczepkowska

## About the project

The aim of this project is to compare efficiency of the following Machine Learning algorithms in picture recognition problem:

- K-Nearest Neighbors (KNN)
- Neural Network
- Convolutional Neural Network

## About the dataset

The dataset contains around 25k images of Natural Scenes around the world.

All training and testing images with a size of 150x150 are classified into 6 categories:

- buildings = 0
- forest = 1
- glacier = 2
- mountains = 3
- sea = 4
- street = 5

The data consists of 2 separated datasets - for training and testing the models.

## How to run this project

1. Clone the git repository
2. Enter the folder Project-PZND
3. Download the data
```dvc pull```
4. Create virtual environment
```python -m venv venv```
5. Activate virtual environment
from PyCharm: ```.\venv\Scripts\activate```
from terminal: ```source venv/Scripts/activate```
6. Install the necessary packages
```pip install -r requirements.txt```
7. Run the project
```cd scripts```
```python main.py```
