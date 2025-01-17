# Import necessary PySpark modules for creating a Spark session, handling DataFrame operations,
# and specifying data types.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, stddev, corr, lit
from pyspark.sql.types import IntegerType, FloatType, StringType

# Function to create a Spark session
def create_spark_session():
    """
    Create and return a SparkSession object.
    This is the entry point for using PySpark.
    """
    return SparkSession.builder \
        .appName("Heart Disease Analysis") \
        .getOrCreate()  # Initialize or retrieve an existing SparkSession

# Define the file path for the input dataset
file_path = "heart.csv"

# Function to load the dataset
def load_data(spark, file_path):
    """
    Load the dataset into a PySpark DataFrame.
    Parameters:
        spark (SparkSession): The active Spark session.
        file_path (str): Path to the CSV file to be loaded.
    Returns:
        DataFrame: Loaded dataset.
    """
    try:
        # Read the CSV file into a DataFrame with headers and automatic schema inference
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Function to clean the data
def clean_data(df):
    """
    Clean the dataset by handling missing values and ensuring consistent data types.
    Parameters:
        df (DataFrame): The input DataFrame.
    Returns:
        DataFrame: Cleaned DataFrame.
    """
    # Drop rows with any missing values
    df = df.dropna()

    # Cast columns to appropriate data types for consistency
    df = df.withColumn("age", col("age").cast(IntegerType())) \
           .withColumn("chol", col("chol").cast(FloatType())) \
           .withColumn("thalach", col("thalach").cast(FloatType())) \
           .withColumn("sex", col("sex").cast(IntegerType())) \
           .withColumn("target", col("target").cast(IntegerType()))
    return df

# Function to create derived features
def create_derived_features(df):
    """
    Add derived features such as age groups for analysis.
    Parameters:
        df (DataFrame): The input DataFrame.
    Returns:
        DataFrame: DataFrame with additional derived features.
    """
    # Define age groups using conditional expressions
    age_groups = when(col("age") < 30, "20-29") \
                 .when((col("age") >= 30) & (col("age") < 40), "30-39") \
                 .when((col("age") >= 40) & (col("age") < 50), "40-49") \
                 .when((col("age") >= 50) & (col("age") < 60), "50-59") \
                 .when((col("age") >= 60), "60+")
    
    # Add a new column for age groups
    df = df.withColumn("age_group", age_groups)
    return df

# Function to analyze distribution of heart disease
def analyze_distribution(df):
    """
    Calculate the distribution of heart disease by age groups and gender.
    Parameters:
        df (DataFrame): The input DataFrame.
    Returns:
        DataFrame: Distribution summary DataFrame.
    """
    distribution = df.groupBy("age_group", "sex").agg(
        count(when(col("target") == 1, 1)).alias("heart_disease_count"),  # Count rows where target=1
        count("*").alias("total_count")  # Count all rows
    )
    return distribution

# Function to identify significant factors correlated with heart disease
def identify_significant_factors(df):
    """
    Identify the top 3 numeric features most correlated with heart disease.
    Parameters:
        df (DataFrame): The input DataFrame.
    Returns:
        list: List of top 3 features with their correlation values.
    """
    # List of numeric columns to evaluate
    numeric_cols = ["age", "chol", "thalach"]
    correlations = []

    # Compute correlation of each feature with the target column
    for col_name in numeric_cols:
        corr_value = df.stat.corr(col_name, "target")
        correlations.append((col_name, abs(corr_value)))  # Store absolute correlation value

    # Sort features by correlation value in descending order and select the top 3
    top_factors = sorted(correlations, key=lambda x: x[1], reverse=True)[:3]
    return top_factors

# Function to analyze chest pain patterns
def analyze_chest_pain_patterns(df):
    """
    Analyze patterns between chest pain types (cp) and heart disease presence.
    Parameters:
        df (DataFrame): The input DataFrame.
    Returns:
        DataFrame: Summary DataFrame with chest pain patterns.
    """
    patterns = df.groupBy("cp").agg(
        count(when(col("target") == 1, 1)).alias("heart_disease_count"),  # Count rows where target=1
        count("*").alias("total_count")  # Count all rows
    )
    return patterns

# Function to save DataFrame as Parquet
def save_as_parquet(df, output_path):
    """
    Save the DataFrame as a Parquet file.
    Parameters:
        df (DataFrame): The DataFrame to save.
        output_path (str): The directory to save the Parquet file.
    """
    try:
        # Write DataFrame to Parquet format with overwrite mode
        df.write.mode("overwrite").parquet(output_path)
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

# Main function to execute the script
def main():
    # Create a Spark session
    spark = create_spark_session()

    # Define file paths
    input_file = file_path
    output_path_cleaned = "df_cleaned"  # Directory for cleaned data
    output_path_analysis = "df_analysis"  # Directory for analysis results

    # Load and clean data
    df = load_data(spark, input_file)
    df_cleaned = clean_data(df)
    df_features = create_derived_features(df_cleaned)

    # Perform analysis
    distribution = analyze_distribution(df_features)
    top_factors = identify_significant_factors(df_features)
    chest_pain_patterns = analyze_chest_pain_patterns(df_features)

    # Save results as Parquet
    save_as_parquet(df_cleaned, output_path_cleaned)
    save_as_parquet(distribution, f"{output_path_analysis}/distribution")
    save_as_parquet(chest_pain_patterns, f"{output_path_analysis}/chest_pain_patterns")

    # Print results of top factors
    print("Top 3 significant factors:")
    for factor, corr in top_factors:
        print(f"{factor}: {corr}")

    # Stop the Spark session
    spark.stop()

# Entry point for the script
if __name__ == "__main__":
    main()