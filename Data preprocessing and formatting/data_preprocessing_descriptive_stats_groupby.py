#loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
import seaborn as sb
from scipy import stats


url_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(url_path, header=None)


df.replace("?", np.nan, inplace = True)
#view the dataset
print(df.head())
print(df.tail())
#adding column names to the dataset
df.columns=['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels',
                            'engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size',
                            'fuel-system','bore','stroke','compression-ratio','horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'];


#descriptive statistics
print(df.describe(include='all'))
print(df.describe())
df.describe(include=['object'])

#analysing drive wheel count
drive_wheel_count = df["drive-wheels"].value_counts()
drive_wheel_count.rename(columns={'drive-wheels': 'drive_wheel_count'}, inplace=True)
drive_wheel_count.index.name = 'drive-wheels-count'
print("Drive wheel count",drive_wheel_count.head())

#addressing missing data values
mean_price = df["price"].astype(float).mean()
print(mean_price)
df["price"].replace(np.nan, mean_price,inplace=True)
print(df["price"])

mean_peak_rpm = df["peak-rpm"].astype(float).mean()
print(mean_peak_rpm)
df["peak-rpm"].replace(np.nan, mean_peak_rpm,inplace=True)
print(df["peak-rpm"])

#Converting to required data type
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

#finding relationship between two columns in a dataframe
#x axis : predictive variable/independent variable
#y axis : target variable
x = df["engine-size"]
y = df["price"]
mplt.pyplot.scatter(x,y)
mplt.pyplot.xlabel("engine size")
mplt.pyplot.ylabel("price")
mplt.pyplot.legend()
mplt.pyplot.title("Relationship between engine size and price: Scatter Plot")
pyplot.show()

#importing seaborn to add linear line to further analyse
#analysis: Positive linear relationship (has a correlation in both varible) x increases and y also increases
sb.regplot(x,y)
mplt.pyplot.title("Relationship between engine size and price: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()

print("Correlation value for engine size and price",df[["engine-size", "price"]].corr())

#analysis: negative linear relationship  since the slope of the line is steep. x increases and y dccreases
sb.regplot(x='highway-mpg',y='price', data= df)
mplt.pyplot.title("Relationship between highway-mpg and price: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()

print("Correlation value for highway-mpg and price",df[["highway-mpg", "price"]].corr())

#analysis: no linear relationship  and no correlation relationship because we cannot find any pattern of x or y increase or decrease
sb.regplot(x='peak-rpm',y='price', data= df)
mplt.pyplot.title("Relationship between highway-mpg and price: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()

print("Correlation value for peak-rpm and price",df[["peak-rpm", "price"]].corr())

# Analyse correlation usinf Pearson correlation with correlation coefficient and p-value
# Correlation coefficinet: Close to +1 : Positive relationship ; close to -1: Negative relationship; 0+ no relationship
# p-value: Strong certainty : p < 0.001 , Moderate : p < 0.05 , weak : p <0.1 , No certainty : p > 0.1
correlation_coefficinet, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("correlation coefficient",correlation_coefficinet,"and p-value",p_value)

# ANOVA : To find correlation among categorical variable (Analysis of Variance)
# F test : based on the difference in price segment of each category for particular make
# p-value : confidence degree
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.get_group('4wd')['price']
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

#group by function to be done on catwgorial variable
df_test_data = df[['drive-wheels','body-style','price']]
grouped_data = df_test_data.groupby(["drive-wheels","body-style"],as_index=False).mean()
df_pivot = grouped_data.pivot(index="drive-wheels",columns="body-style")
grouped_pivot = df_pivot.fillna(0) #fill missing values with 0
#Heat map
mplt.pyplot.pcolor(df_pivot)
mplt.pyplot.colorbar()
mplt.pyplot.title("Heat map of grouped data")
mplt.pyplot.legend()
pyplot.show()

df_test_data1 = df[['body-style','price']]
grouped_data_body_style = df_test_data1.groupby(["body-style"],as_index=False).mean()
df_pivot1 = grouped_data_body_style.pivot(index="price",columns="body-style")
#Heat map
mplt.pyplot.pcolor(df_pivot1)
mplt.pyplot.colorbar()
mplt.pyplot.title("Heat map of grouped data body style and price")
mplt.pyplot.legend()
pyplot.show()

#find correlation using corr() method
print(df.corr())
#correlation for specific coulmn
df_test = df[['bore', 'stroke', 'compression-ratio', 'horsepower']]
print(df_test.corr())

#scatter plot for stroke and price
mean_stroke = df["stroke"].astype(float).mean()
print(mean_stroke)
df["stroke"].replace(np.nan, mean_stroke,inplace=True)
print(df["stroke"])

df[["stroke"]] = df[["stroke"]].astype("float")

sb.regplot(x="stroke",y="price", data=df)
mplt.pyplot.title("Relationship between stroke and price: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()

print("Correlation value for stroke and price",df[["stroke", "price"]].corr())

#Boxplot using seaborn (Categorical variables can be represented in box plot)
sb.boxplot(x="engine-location", y="price", data = df)
mplt.pyplot.title("Box plot of engine location and price")
pyplot.show()

sb.boxplot(x="drive-wheels", y="price", data = df)
mplt.pyplot.title("Box plot of drive wheels and price")
pyplot.show()