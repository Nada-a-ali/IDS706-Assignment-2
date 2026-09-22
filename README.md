# 1. Coffee Quality Data Analysis

In this project, a coffee quality dataset was imported via Kaggle, linking authored Git <https://github.com/jldbc/coffee-quality-database.git> for analysis. This dataset contains information from reviewers for Arabica and Robusta coffee bean types. 

The imported dataset includes quality measures (e.g., flavor, aftertaste, sweetness) as well as farm metadata (e.g., country of origin, region), among other information. 

# Data Analysis Steps 

## 2. Inspecting the Data 

To inspect the dataset, imported libraries (e.g., Pandas, Numpy) were used to visualize overview summary statistics such as column names, data range, and data types. 

### Brief Description of Imported Data 

The imported data contains 1339 entries and 44 columns with float, integer, and string data types. Duplicate / missing value checks revealed missing values common among column coffee farm names (n=359), lot numbers (n=1063), and producers (n=232).  

## 3. Basic Filtering and Grouping / Visualization Bar Type 

Filters were then extracted for meaningful subsets and summary statistics:
- Type of coffee bean (i.e., Arabica or Robusta): 28 Robusta coffee bean types included versus 1311 Arabica coffee bean types included in the following dataset. 
- Coffee cups scored above average: Using the average coffee quality reviewer score indicated by "Total.Cup.Points" in the descriptive statistics populated prior, then further filtered by coffee type (i.e., Arabica or Robusta)
- Cups, filtered by type of coffee, scored above average by country of origin [i.e., .groupby() function]. 

After filtering, produced separate simple bar charts of Arabica and Robusta coffee types scored above average (as indicated by "Total.Cup.Points") by country of origin. 

## 4. Exploring a Machine Learning Algorithm 

Linear regression was implemented to begin experimenting with model inputs and outputs. Specifically, to explore whether "sweetness" can be used to predict coffee quality rating as well as exploring whether "flavor" and "aftertaste" can be used to predict coffee quality rating (quality indicated via "Total.Cup.Points"). 

# 5. Experimenting with Rust 

Ran the Rust Jupyter Notebook and changed variable assignment to begin experimenting with its functionality. Rust contains stricter rules as compared with Python. Ownership and variable assignment differ, with Rust forcing ownership, borrowing, and synchronization rules to be explicit. 

# Testing 

## 1. Unit Tests 

Unit tests check for data loading and dataset structure, handling of missing file, filtering coffee with above average quality scores, filtering coffee type ("Robusta" or "Arabica), and sweetness-based linear regression predictions. 

## 2. System / Integration Tests 

End-to-end test implementation to check overall workflow by loading the dataset, applying the coffee-quality filters, training the sweetness linear regression model as a predictor for coffee quality, and confirming the results are produced successfully. 

## 3. Test Pass 

Pytest used for automated testing. Run tests with: 

python3 -m pytest -v


