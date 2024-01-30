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

### Setting up virtual environment and installing requirements

After cloning the repository set up virtual environment and install requirements:

```bash
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

### Running the project
Now all that is left to do is running main.py.