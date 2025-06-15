import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


### loading dataset
file_path = '/Users/srikantchamarthi/Documents/AI/ECR Process/simulated_ecr_dataset.csv'
df = pd.read_csv(file_path)
#print(df.head())
#print(df.tail())

'''
# Convert date columns to datetime format
df['Date_Raised'] = pd.to_datetime(df['Date_Raised'], errors='coerce')

# Check for missing values in each column
print(df.isna().sum())

# Handle or fill missing values in columns other than Effectiveness_Rating first

# turnadround days have the Nan's and filling it with Approval_Days + Implementation_Days
df['Total_Turnaround'] = df['Total_Turnaround'].fillna(df['Approval_Days'] + df['Implementation_Days'])

# Encode categorical columns for modeling
categorical_cols = ['Change_Type', 'Request_Department', 'Assigned_To', 'Priority', 'Status', 'Delay_Cause']

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

#print(df.head())

######## preparing the training sets 

feature_cols = ['Change_Type', 'Request_Department', 'Assigned_To', 'Priority', 'Status', 'Approval_Days', 'Implementation_Days','Total_Turnaround', 'Delay_Cause']

##separating rows with and without effective ratings

train_df = df[df['Effectiveness_Rating'].notna()]
test_df =  df[df['Effectiveness_Rating'].isna()]

X_train = train_df[feature_cols]
Y_train = train_df['Effectiveness_Rating']


#checkign the model accuracy using cross-validation on the rows where effectiveness rating is known and to see how the model performs when predicting the values fo the empty cells in the effective rating. 

### cross validation (10folds)

model= RandomForestRegressor(random_state=42)

## R2 score
r2_score = cross_val_score(model, X_train, Y_train, cv=10)
print(f"cross_valdiation score:{r2_score}")
print(f"average r2 score: {r2_score.mean():.3f}")
'''

##R2 score is worse than using the mean values for the effecitveness ratinng. Also, as the ecr stage is reached only when the PR is diusccswed and an appropriate solution is availbale. So, dropping the effectiveess rating column

## dorpping the effectiveness rating column
df = df.drop(columns=['Effectiveness_Rating'], errors='ignore')

###convert date and compute total turnaround if missing values
df['Date_Raised'] = pd.to_datetime(df['Date_Raised'], errors= 'coerce')
df['Total_Turnaround'] = df['Total_Turnaround'].fillna(df['Approval_Days'] + df['Implementation_Days'])

### ecnoding categorical values (basically converting non numerical data to numeric foirm)

categorical_cols = ['Change_Type', 'Request_Department', 'Assigned_To', 'Priority', 'Status', 'Delay_Cause']
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

print(df.head())
