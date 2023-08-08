

import pandas as pd
import numpy as np

data1= pd.read_json('Order_breakdown.json')
print(data1.shape)
data1.head()

data2= pd.read_csv('Order.tsv',sep='\t')
print(data2.shape)
data2.head()

df =pd.merge(data1,data2,on='Order ID')

df.to_csv('df')

print(df.isnull().sum())
print('total duplicated before',df.duplicated().sum())
df.drop_duplicates(inplace= True)
print('total duplicated after',df.duplicated().sum())
print(df.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# creating a for loop to check unique values of all the columns if values of the unique value is less than 10 we will print value count of
#unique values too else we will just show how many unique values are there.

for i in df.columns:
    # Check if the number of unique values in the column is less than 10

    if df[i].nunique() < 10:
        # Print column information, including the column name, data type, number of unique values, and value counts
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values: \n{df[i].value_counts()}')
        # Print a separator line
        print(40*'_')
    else:
        # Print column information, including the column name, data type, and number of unique values
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values')
        # Print a separator line
        print(40*'_')

dfc=df[[ 'Category', 'Sub-Category',
       'Country', 'Region', 'Segment',
       'Ship Mode']]

x = 0
fig = plt.figure(figsize=(20, 25))  # Create a figure with the specified size
plt.subplots_adjust(wspace=0.4)  # Adjust the spacing between subplots

# Iterate over each column in the DataFrame dfc
for i in dfc.columns:
    ax = plt.subplot(421 + x)  # Create a subplot at position 321+x in a 3x2 grid
    ax = sns.countplot(data=dfc, y=i)  # Create a countplot using data from dfc with y-axis representing the column values
    plt.grid(axis='x')  # Add gridlines along the x-axis
    ax.set_title(f'Distribution of {i}')  # Set the title of the subplot
    x += 1  # Increment x to move to the next subplot position

"""most sold category is office supplies

1.   most sold category is office supplies.
2.   top 3 in terms of orders are uk,germany and france is no1.
3.   Most of the sales are from Consumer segment
4.   Most used ship mode in Economy




"""

df['Margin']= df['Profit']/df['Sales'] *100

df.head()

a =df.groupby(['Country'])[['Sales','Margin']].mean()

plt.figure(figsize=(20, 15))
a.plot(kind='bar')

"""
**denmark, Ireland, the Netherlands, Portugal, and Sweden are experiencing negative margins. The countplot analysis indicates that the Netherlands and Sweden have relatively higher quantities of sales, but upon examining their mean sales, it becomes evident that they are considerably low. This suggests that in their efforts to increase order quantity, these countries have compromised their selling prices.**"""

plt.figure(figsize=(16, 10))  # Set the size of the entire figure

# First subplot (left side)
plt.subplot(2, 1,1)
sns.barplot(data=df, x='Country', y='Margin',hue='Category')
plt.title('Margin')
plt.legend()

# Second subplot (right side)
plt.subplot(2,1, 2)
sns.barplot(data=df, x='Country', y='Sales',hue='Category')
plt.title('Sales')
plt.legend()
plt.show()

"""Sweden, Denmark, Portugal, Ireland, and the Netherlands are all
experiencing negative margins across all three categories: office supplies, furniture, and technology.

Among these countries, the furniture category seems to be the most significant contributor to the negative margins in most cases.

The Netherlands is an exception, where the technology category has a higher negative impact on Margins compared to furniture.

Surprisingly, even Italy has recorded negative margins in both technology and furniture; however, it managed to achieve positive margins in the office supplies category.

"""
df['Order Date'] = pd.to_datetime(df['Order Date']) 
df['Month'] = pd.to_datetime(df['Order Date']).dt.strftime('%B')
df['Year'] = pd.to_datetime(df['Order Date']).dt.year

plt.figure(figsize=(20, 10))

sns.countplot(x='Sub-Category', hue='Year', data=df)
plt.xlabel('Sub-Category')
plt.ylabel('Count')
plt.title('Countplot of Sub-Category with Margin')
plt.show()

"""Sales data reveals that art storage and binders have experienced a notable upsurge in their sales count, whereas tables are witnessing comparatively lower sales figures. This discrepancy could potentially be because of  the pricing dynamics."""

df['single Quantity price']= df['Sales']/df['Quantity']

plt.figure(figsize=(20, 10))
plt.subplot(2,1, 2)
sns.barplot(data=df,y='single Quantity price',x='Sub-Category',hue='Year')
plt.title('Sales')
plt.legend()
plt.show()

"""above chart proves that sales count is dependent on sale price"""

plt.figure(figsize=(25, 14))
ax= plt.subplot(2,1,2)
sns.barplot(x=df['Sub-Category'], y=df['Margin'],hue= df['Region'])

ax= plt.subplot(2,1,1)
sns.barplot(x=df['Sub-Category'], y=df['Sales'], hue=df['Region'])

"""In all three regions, North experienced the highest sales of tables. Surprisingly, despite the higher sales, the data indicates that North managed to sell tables at a negative margin, although it was close to achieving positive margins.

On the other hand, the situation was quite different in the Central and South regions. Despite having higher sales of tables, both regions suffered significantly in terms of profits. The data suggests that the margins in these regions were considerably negative, leading to substantial losses even with the higher sales volume.
"""

fig = plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
df.groupby("Region")["Sales"].sum().plot.pie(autopct = "%1.0f%%")
plt.title("Region Vs Sales")
plt.subplot(1, 2, 2)
df.groupby("Region")["Profit"].sum().plot.pie(autopct = "%1.0f%%")
plt.title("Region vs Profit")
plt.show()

plt.figure(figsize=(25, 14))
ax=plt.subplot(2,1,1)
sns.barplot(x=df['Sub-Category'], y=df['Sales'],errorbar=None)

ax=plt.subplot(2,1,2)
sns.barplot(x=df['Sub-Category'], y=df['Discount'],errorbar=None)
plt.show()

"""High discounts boosted sales of tables and chairs. Bookcases, appliances, and copiers saw good sales with 10-15% discounts. However, storage and fasteners struggled to generate significant sales despite offering substantial discounts."""

plt.figure(figsize=(25, 14))
ax=plt.subplot(2,1,1)
sns.barplot(x=df['Month'], y=df['Sales'],hue=df['Year'],errorbar=None)

ax=plt.subplot(2,1,2)
sns.barplot(x=df['Month'], y=df['Margin'],errorbar=None,hue=df['Year'])
plt.show()

"""lowest margin- january 2013
highest margin-may 2014
"""

df13=df[df['Year']==2013]
df14=df[df['Year']==2014]
df15=df[df['Year']==2015]
df16=df[df['Year']==2016]

plt.figure(figsize=(25, 50))


ax = plt.subplot(8, 1, 1)
sns.barplot(x='Month', y='Sales',hue='Category' ,data=df13, errorbar=None)
ax = plt.subplot(8, 1, 2)
sns.barplot(x='Month', y='Margin', hue= 'Category',  data=df13, errorbar=None)
plt.title('2013 margin and sales')


ax = plt.subplot(8, 1, 3)
sns.barplot(x='Month', y='Sales',hue='Category' ,data=df14, errorbar=None)

ax = plt.subplot(8, 1, 4)
sns.barplot(x='Month', y='Margin', hue= 'Category',  data=df14, errorbar=None)
plt.title('2014 margin and sales')

ax = plt.subplot(8, 1, 5)
sns.barplot(x='Month', y='Sales',hue='Category' ,data=df15, errorbar=None)
ax = plt.subplot(8, 1, 6)
sns.barplot(x='Month', y='Margin', hue= 'Category',  data=df15, errorbar=None)
plt.title('2015 margin and sales')


ax = plt.subplot(8, 1, 7)
sns.barplot(x='Month', y='Sales',hue='Category' ,data=df16, errorbar=None)
ax = plt.subplot(8, 1, 8)
sns.barplot(x='Month', y='Margin', hue= 'Category',  data=df16, errorbar=None)
plt.title('2016 margin and sales')
plt.show()

"""
In 2013, negative margins were observed in January, March, and April, specifically in the furniture category.

In 2014, negative margins occurred in January, March, April, June, and November, also in the furniture category.

In 2015, negative margins were recorded in January and March, with very low negative margins in August and September, still within the furniture category.

In 2016, negative margins were observed in February, May, June, and December, again in the furniture category.

The negative margins seem to occur intermittently throughout each year, with specific months being more prone to experiencing these challenges.
This suggests that there might be seasonality or cyclical factors influencing the furniture sales and profitability."""

plt.figure(figsize=(25, 14))

# Subplot 1: Sales with hue based on both Year and Category
ax = plt.subplot(2, 1, 1)
sns.barplot(x='Sub-Category', y='Sales',hue='Year' ,data=df, errorbar=None)

# Subplot 2: Margin with hue based on both Year and Category
ax = plt.subplot(2, 1, 2)
sns.barplot(x='Sub-Category', y='Margin', hue= 'Year',  data=df, errorbar=None)

plt.show()

"""The chart demonstrates that the tables category has significantly improved its margins over time. In 2014, the margins were very poor, but there has been a remarkable reversal since then. The improved margins have been accompanied by a notable increase in sales figures, indicating a positive correlation between margin improvement and sales volume growth.


On the other hand, the chairs segment has been experiencing a negative trend in margins year after year. In 2013, margins were positive, but they gradually decreased in subsequent years.However sales volume are also increasing see countplot chart below
"""

plt.figure(figsize=(16, 5))

ax=plt.subplot(1,2,1)
sns.countplot(x='Sub-Category', data=df[df['Sub-Category'] == 'Tables'],hue='Year')
plt.title('Countplot for Tables')

ax=plt.subplot(1,2,2)
sns.countplot(x='Sub-Category', data=df[df['Sub-Category'] == 'Chairs'],hue='Year')
plt.title('Countplot for Chairs')

plt.figure(figsize=(25, 12))

ax = plt.subplot(2, 1, 1)
sns.barplot(x='Quantity', y='Actual Discount',hue='Category',data=df, errorbar=None)

ax = plt.subplot(2, 1, 2)
sns.barplot(x='Quantity', y='Margin',hue='Sub-Category',data=df[df['Category'] == 'Furniture'], errorbar=None)
plt.show()



df['Sub-Category'].value_counts()

plt.figure(figsize=(16, 5))


ax=plt.subplot(1,2,1)
sns.barplot(x='Days to Ship', y='Profit',hue='Category' , data=df, errorbar=None)

ax=plt.subplot(1,2,2)
sns.barplot(x='Days to Ship', y='Sales',hue='Category' , data=df, errorbar=None)
plt.show()

df['State'].value_counts()

top10cities=df.groupby(['City','Country'])[['Sales','Margin']].mean()
top10cities=top10cities.sort_values(by='Sales',ascending=False)
top10cities= top10cities.head(10)

bottom10cities=df.groupby(['City','Country'])[['Sales','Margin']].mean()
bottom10cities=bottom10cities.sort_values(by='Sales',ascending=True)
bottom10cities= bottom10cities.head(10)

bottom10cities.plot(kind='bar', y=['Sales', 'Margin'], figsize=(12, 4))

top10cities.plot(kind='bar', y=['Sales', 'Margin'], figsize=(12, 4))

top10clients=df.groupby(['Customer Name'])[['Sales','Margin']].mean()
top10clients=top10clients.sort_values(by='Sales',ascending=False)
top10clients= top10clients.head(10)

bottom10clients=df.groupby(['Customer Name'])[['Sales','Margin']].mean()
bottom10clients=bottom10clients.sort_values(by='Sales',ascending=True)
bottom10clients= bottom10clients.head(10)

bottom10clients.plot(kind='bar', y=['Sales', 'Margin'], figsize=(12, 4))

""" the bottom 10 clients, even though they buy less, are helping us make more profit because they're buying expensive things or services that give us a lot of profit.  
 Eleanor, Kiab, and Jacob got big discounts but didn't buy much.

On the other hand, the top 10 clients are buying a lot, which makes our overall sales look good, but they're buying things that don't give us as much profit each time. However, since they buy so much, their total impact on our sales is big.
"""

top10clients.plot(kind='bar', y=['Sales', 'Margin'], figsize=(12, 4))

plt.figure(figsize = (10, 5))
# profit/loss by Discount level
sns.lineplot(x='Discount', y= 'Profit', data = df, color = 'Teal', label = 'Discount Level')
plt.ylabel('Profit/Loss in USD$')
plt.title('Profit/Loss by Discount Level', fontsize = 20)
plt.show()

#corr=df.corr()

#sns.heatmap(corr,annot= True)

df.columns

df.drop(['Margin','Year','Month','Customer Name','Order ID','Order Date','Ship Date','single Quantity price'],axis=1,inplace= True)

dfif= df.select_dtypes(include=['int','float'])

dfif.dtypes

x=0
plt.figure(figsize=(25,25))
for i in dfif.columns:
  ax= plt.subplot(321+x)
  sns.boxplot(dfif[i])
  plt.title(i)
  x+=1

df.describe(percentiles=[0.005,0.01,0.02,0.03,0.95,0.98,0.99,0.996,0.997,0.998,0.999]).T

print(df[df['Discount']>0.6500].shape)
print(df[df['Sales']>3220.8680].shape)
print(df[df['Profit']>1219.6480].shape)
print(df[df['Profit']<-626.34].shape)

df['Sales'] = np.where(df['Sales'] > 3220.8680,3220.8680, df['Sales'])
df['Profit'] = np.where(df['Profit']>1219.6480, 1219.6480, df['Profit'])
df['Profit'] = np.where(df['Profit']<-626.34, -626.34, df['Profit'])
df['Discount'] = np.where(df['Discount'] >0.6500,0.6500	, df['Discount'])

df.describe(percentiles=[0.005,0.01,0.02,0.03,0.95,0.98,0.99,0.996,0.997,0.998,0.999]).T

df['Product Name'].value_counts()

df['Company'] = df['Product Name'].str.split().str[0]

df['Company'].value_counts()

value_count= df['Company'].value_counts()
value_count= value_count[value_count<=10]
print(value_count.sum())
value_to_drop= list(value_count.index)
df.drop(df[df["Company"].isin(value_to_drop)].index,axis=0,inplace = True)

df['State'].value_counts()

value_count= df['State'].value_counts()
value_count= value_count[value_count<10]
print(value_count.sum())
rows_to_drop=list(value_count.index)
df.drop(df[df['State'].isin(rows_to_drop)].index,axis=0,inplace= True)

df.drop(['Product Name'],axis=1,inplace= True)

df.drop(['Actual Discount'],axis=1,inplace= True)

df.drop(['City'],axis=1,inplace= True)

from sklearn.model_selection import train_test_split

x= df.drop('Profit',axis=1)
y= df['Profit']

xtrain,xtest,ytrain,ytest= train_test_split(x,y,random_state=25,test_size=0.25)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

d= {'model':[],'mse':[],'rmse':[],'mae':[],'r2s':[]}
def model_eval(model,ytest,ypred):
  mse= mean_squared_error(ytest,ypred)
  mae= mean_absolute_error(ytest,ypred)
  rmse= np.sqrt(mse)
  r2s= r2_score(ytest,ypred)
  print('mse:', mse)
  print('rmse:', rmse)
  print('r2s:', r2s)
  print('mae:', mae)
  d['model'].append(model)
  d['mse'].append(mse)
  d['rmse'].append(rmse)
  d['mae'].append(mae)
  d['r2s'].append(r2s)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df.head()

xtrain.dtypes

# Step 1: ColumnTransformer for categorical feature encoding and passthrough

step1= ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[3,4,5,6,7,8,9,11])],
                         remainder='passthrough')
# Step 2: Regression model

step2 = LinearRegression()
pipelr= Pipeline([('step1',step1),('step2',step2)])

pipelr.fit(xtrain,ytrain)
ypredlr=pipelr.predict(xtest)

model_eval('regression',ytest,ypredlr)

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [4,3,5,6,7,8,9,11])],
                          remainder='passthrough')

# Step 2: Ridge Regression model with alpha=2.1
step2 = Ridge(alpha=2.1)

# Create a pipeline with the defined steps
piperid = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline to the training data
piperid.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredrid = piperid.predict(xtest)

# Evaluate the performance of the Ridge Regression model
model_eval('Ridge', ytest, ypredrid)
# This step calls the 'model_eval' function to evaluate the performance of the Ridge Regression model.

step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', sparse=False), [4,3,5,6,7,8,9,11])],
                          remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=250, max_depth=7, min_samples_split=5, random_state=14)

# Create the pipeline by combining the preprocessing steps and the model
piperf = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
piperf.fit(xtrain, ytrain)

# Make predictions on the test data
ypredrf = piperf.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('rf', ytest, ypredrf)

step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False),[4,3,5,6,7,8,9,11])],
                          remainder='passthrough')

# Step 2: AdaBoost Regression with RandomForestRegressor as the base estimator
step2 = AdaBoostRegressor(RandomForestRegressor(n_estimators=25, max_depth=7, min_samples_split=7, random_state=14),
                          n_estimators=5)


# Create a pipeline with the defined steps
pipeadar = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and AdaBoost Regression into a single object.

# Fit the pipeline to the training data
pipeadar.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredadar = pipeadar.predict(xtest)

# Evaluate the performance of the AdaBoost Regression with RandomForest model
model_eval('adarf', ytest, ypredadar)

from sklearn.ensemble import BaggingRegressor
# Define the preprocessing steps and the BaggingRegressor with DecisionTreeRegressor base estimator
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [3,4,5,6,7,8,9,11])],
                          remainder='passthrough')
step2 = BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=9, random_state=5),
                         n_estimators=15, max_samples=xtrain.shape[0], max_features=xtrain.shape[1], random_state=2022)

# Create the pipeline by combining the preprocessing steps and the model
pipebrdt = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
pipebrdt.fit(xtrain, ytrain)

# Make predictions on the test data
ypredbrdt = pipebrdt.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('bgdt', ytest, ypredbrdt)

step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', sparse=False), [3,4,5,6,7,8,9,11])],
                          remainder='passthrough')
step2 = DecisionTreeRegressor(max_depth=13, min_samples_split=9, random_state=5)

# Create the pipeline by combining the preprocessing steps and the model
pipedt = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
pipedt.fit(xtrain, ytrain)

# Make predictions on the test data
ypreddt = pipedt.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('dt', ytest, ypreddt)

from sklearn.ensemble import GradientBoostingRegressor
# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=False), [3,4,5,6,7,8,9,11])],
                          remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=10,  max_depth=55, random_state=42)

# Create a pipeline with the defined steps
pipegb = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline to the training data
pipegb.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredgb = pipegb.predict(xtest)

# Evaluate the performance of the Gradient Boosting Regression model
model_eval('gb', ytest, ypredgb)

from sklearn.linear_model import HuberRegressor
# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=False), [3,4,5,6,7,8,9,11])],
                          remainder='passthrough')

step2 = HuberRegressor(epsilon=5,max_iter=10,alpha=1)

# Create a pipeline with the defined steps
pipehr = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline to the training data
pipehr.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredgb = pipehr.predict(xtest)

# Evaluate the performance of the Gradient Boosting Regression model
model_eval('hr', ytest, ypredgb)


d



data= pd.DataFrame(d)

data

df.columns

df.head()

import pickle

pickle.dump(piperf,open('piperf.pkl','wb'))
pickle.dump(xtrain,open('ordermain.pkl','wb'))

model=pickle.load(open('piperf.pkl','rb'))

type(model)

random20= df.sample(20)

random20

prediction= model.predict(random20)

random20['prediction']= prediction

random20.head()

random20=random20[['Profit','prediction']]

random20

