# K-means Clustering Implementation

This project provides a custom implementation of the K-means clustering algorithm, along with tools for data generation, visualization, and comparison with scikit-learn's K-means implementation.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [License](#license)

## Features

- Custom K-means clustering implementation
- K-means++ initialization
- Elbow method for optimal k selection
- Custom dataset generation
- Visualization tools for clusters and elbow method
- Comparison with scikit-learn's K-means implementation

## Project Structure

The project is organized into several Python modules:

- `kmeans.py`: Core K-means algorithm implementation
- `data_generation.py`: Functions for generating custom datasets
- `visualization.py`: Plotting functions for clusters and elbow method
- `analysis.py`: Functions for comparing K-means implementations and running the elbow method
- `main.py`: Main script to run the entire analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/kmeans-clustering.git
   cd kmeans-clustering
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.