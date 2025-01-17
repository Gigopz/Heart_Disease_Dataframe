Heart Disease Data Analysis

This project analyzes the Heart Disease Dataset using PySpark. The goal is to clean, preprocess, and analyze the data to uncover insights related to heart disease presence.

Files

heart.csv: The dataset used for analysis.

main.py: The Python script containing the PySpark code for data processing and analysis.

df_cleaned/: Directory containing the cleaned dataset in Parquet format.

df_analysis/: Directory containing analysis results in Parquet format.


Requirements

To run this project, ensure you have the following installed:

Python (3.8 or later)

PySpark (3.5.3)

Apache Hadoop

Java Development Kit (JDK 8 or later)


Approach

Data Loading:

The dataset is loaded into a PySpark DataFrame using spark.read.csv.

Error handling is implemented to manage issues during data loading.


Data Cleaning:

Rows with null values are dropped.

Data types of columns are explicitly cast to ensure consistency.


Feature Engineering:

An "age_group" feature is created based on age ranges to enable group-based analysis.


Analysis;

The distribution of heart disease presence is calculated by age group and gender.

The top 3 factors most correlated with heart disease are identified using correlation coefficients.

Patterns between chest pain types and heart disease presence are analyzed.


Output:

Cleaned data and analysis results are saved in Parquet format for efficient storage and retrieval.


Assumptions:

The dataset contains a header row with column names.

Missing values can be safely dropped without significantly affecting the analysis.

Only numeric columns with potential relevance are considered for correlation analysis.


Usage:

Clone the repository and navigate to the project directory.

Place the heart.csv dataset in the same directory as the script.

Run the script using Python:

python main.py

The cleaned data and analysis results will be saved in the specified output directories.


Acknowledgments:

The dataset is sourced from Kaggle's Heart Disease Dataset.
