import pandas as pd

# Load the dataset
file_path_attrition = r'C:\Users\IoannisZografakis-Re\Downloads\employee_attrition.csv'  # Adjust this file path to where your file is located
attrition_df = pd.read_csv(file_path_attrition)

# Step 1: Apply filters
filtered_attrition_df = attrition_df[
    (attrition_df['Attrition'] == 'Yes') &
    (attrition_df['YearsAtCompany'] < 10) &
    (attrition_df['MonthlyIncome'].between(2000, 15000)) &
    (attrition_df['JobRole'].isin(['Sales Executive', 'Research Scientist', 'Laboratory Technician'])) &
    (attrition_df['Age'].between(25, 45)) &
    (attrition_df['PerformanceRating'] >= 3)
]

# Step 2: Set a valid output path (adjust this path to your local system, e.g., Documents folder)
output_file_path_attrition = r'C:\Users\IoannisZografakis-Re\Documents\filtered_employee_attrition.csv'

# Step 3: Export the filtered data to CSV
filtered_attrition_df.to_csv(output_file_path_attrition, index=False)

# Optional: Print the path of the saved file
print(f"Filtered data has been saved to {output_file_path_attrition}")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Select the relevant numeric columns
numeric_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Extract the numeric data
numeric_data = attrition_df[numeric_columns]

# Handle any missing values by filling them with the mean of the respective columns
numeric_data.fillna(numeric_data.mean(), inplace=True)

# Calculate the correlation matrix using NumPy
correlation_matrix = numeric_data.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Customize the heatmap
plt.title('Correlation Heatmap of Selected Employee Attributes')
plt.tight_layout()
plt.show()


# Step 1: Filter the data based on JobRole and Department
filtered_df = attrition_df[
    (attrition_df['JobRole'].isin(['Sales Executive', 'Research Scientist', 'Laboratory Technician'])) &
    (attrition_df['Department'].isin(['Sales', 'Research & Development']))
]

# Step 2: Group the data by YearsAtCompany, JobRole, and Department
grouped_df = filtered_df.groupby(['YearsAtCompany', 'JobRole', 'Department']).agg(
    MeanMonthlyIncome=('MonthlyIncome', 'mean'),
    StdMonthlyIncome=('MonthlyIncome', 'std'),
    EmployeeCount=('MonthlyIncome', 'size')
).reset_index()

# Step 3: Filter out groups with less than 5 employees for reliability
reliable_groups_df = grouped_df[grouped_df['EmployeeCount'] >= 5]

# Step 4: Create a Line Chart with Seaborn to show trends of average monthly income over YearsAtCompany
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create a line chart with markers for data points, and separate lines by Department and JobRole
sns.lineplot(
    data=reliable_groups_df,
    x='YearsAtCompany', y='MeanMonthlyIncome',
    hue='JobRole', style='Department',
    markers=True, dashes=False, err_style=None,
    linewidth=2, marker='o'
)

# Step 5: Add error bars to represent the standard deviation
for i, row in reliable_groups_df.iterrows():
    plt.errorbar(
        x=row['YearsAtCompany'],
        y=row['MeanMonthlyIncome'],
        yerr=row['StdMonthlyIncome'],
        fmt='none', c='gray', capsize=3
    )

# Step 6: Customize the plot
plt.title('Trend of Average Monthly Income over Years at Company', fontsize=14)
plt.xlabel('Years at Company', fontsize=12)
plt.ylabel('Average Monthly Income', fontsize=12)
plt.legend(title='Job Role / Department', fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy import stats

# Step 1: Load the filtered dataset from Task 1
file_path_attrition = r'C:\Users\IoannisZografakis-Re\Documents\filtered_employee_attrition.csv'  # Adjust to the correct path
filtered_attrition_df = pd.read_csv(file_path_attrition)

# Step 2: Filter the relevant columns and remove rows with missing values
filtered_attrition_df = filtered_attrition_df[['Age', 'TotalWorkingYears', 'MonthlyIncome']].dropna()

# Step 3: Perform Multiple Linear Regression (Age, TotalWorkingYears -> MonthlyIncome)
X = filtered_attrition_df[['Age', 'TotalWorkingYears']]
y = filtered_attrition_df['MonthlyIncome']

# Fit the linear regression model
reg_model = LinearRegression()
reg_model.fit(X, y)

# Get the regression coefficients and intercept
coef_age, coef_working_years = reg_model.coef_
intercept = reg_model.intercept_

# Step 4: Generate Grid Values for the regression plane
age_range = np.linspace(filtered_attrition_df['Age'].min(), filtered_attrition_df['Age'].max(), 20)
working_years_range = np.linspace(filtered_attrition_df['TotalWorkingYears'].min(), filtered_attrition_df['TotalWorkingYears'].max(), 20)

age_grid, working_years_grid = np.meshgrid(age_range, working_years_range)
income_grid = intercept + coef_age * age_grid + coef_working_years * working_years_grid

# Step 5: Create 3D Scatter Plot with Plotly
scatter_data = go.Scatter3d(
    x=filtered_attrition_df['Age'],
    y=filtered_attrition_df['TotalWorkingYears'],
    z=filtered_attrition_df['MonthlyIncome'],
    mode='markers',
    marker=dict(
        size=5,
        color=filtered_attrition_df['MonthlyIncome'],  # Color by Monthly Income
        colorscale='Viridis',  # Choose a color scale
        opacity=0.8,
        colorbar=dict(title='Monthly Income')
    ),
    name='Data Points'
)

# Step 6: Add the Regression Plane
plane_data = go.Surface(
    x=age_grid,
    y=working_years_grid,
    z=income_grid,
    colorscale='Reds',
    opacity=0.6,
    showscale=False,
    name='Regression Plane'
)

# Step 7: Customize the Plot and add titles and labels
layout = go.Layout(
    title='3D Scatter Plot with Regression Plane',
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Total Working Years',
        zaxis_title='Monthly Income'
    ),
    autosize=True,
    height=700
)

# Combine the data for scatter plot and regression plane
fig = go.Figure(data=[scatter_data, plane_data], layout=layout)

# Step 8: Show the plot
fig.show()
