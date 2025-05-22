import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#dumpy
#label encoder
#original encoder

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

df_data = pd.read_csv('C:/SONA/car_predict/car_predict/car_price_prediction.csv')

df_data.info()

# =====Build model Machine Learning=====
# Step 1: Tách biến độc lập và biến phụ thuộc (biến mục tiêu ('left'))


#Step 2: Chia dữ liệu thành tập train và tập test

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

encoder = OrdinalEncoder()

'''
Miêu tả dữ liệu

1.   ID	: Mã định danh duy nhất cho mỗi xe trong tập dữ liệu
2.   Price : Giá bán của xe
3. Levy : Thuế hoặc phí bổ sung được áp dụng cho xe
4. Manufacturer : Hãng sản xuất oto
5. Model : Tên cụ thể của kiểu xe
6. Prod. year : Năm sản xuất xe
7. Category : Phân loại xe( SUV, Sedan,...)
8. Leather interior: Nội thất bằng da, cho biết ghế có được bọc da hay không?
9. Fuel type : Loại nhiên liệu, xăng, dầu hybrid
10. Engine volume : Kích thước động cơ, thường được đo bằng lít
11. Mileage : Tổng quãng đường xe đã đi
12. Cylinders : Số xi- lanh của động cơ
13. Gear box type : Loại hộp số, số động hay số sàn
14. Drive wheels : Bánh xe dẫn động( loại dẫn động)- dẫn động cầu trước, cầu sau, dẫn động 4 bánh
15. Doors: Số cửa, chia thành 2-3 , 4-5 , > 5 cửa
16. Wheel: Vị trí vô lăng , trái hoặc phải
17. Color : Màu ngoại thất của xe
18. Airbags : Số lượng túi khí

'''
df_data.duplicated().sum()
print(df_data.isna().sum())
df_data.head()
df_data.select_dtypes(include='object').head()
df_data.select_dtypes(include='object').tail()


df_data['Levy'].isna().sum()

# xac dinh nhung gia tri object bi loi, hoa ra la toan dau "-"
#df_data['Levy']=pd.to_numeric(df_data['Levy'])
invalid_values = df_data[~df_data['Levy'].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna()]
print(invalid_values['Levy'])

#df_data = pd.read_csv('C:/SONA/car_predict/car_predict/car_price_prediction.csv')

# đổi giá trị thuế từ object sang float
df_data['Levy']=pd.to_numeric(df_data['Levy'],errors='coerce')
df_data['Levy'].head()
df_data['Levy'].isna().sum()

print(df_data['Mileage'])
# bo km khoi quang duong
df_data['Mileage'] = df_data['Mileage'].str.replace('km', '', regex=False).str.strip()
df_data['Mileage'].head()

# chuyển dữ liệu object của quãng đường sang dạng int
df_data['Mileage'] = (df_data['Mileage'].astype(int))
df_data['Mileage'].head()

# bỏ chữ Turbo khỏi dung tích động cơ
df_data['Engine volume'] = df_data['Engine volume'].str.replace('Turbo', '', regex=False).str.strip()
df_data['Engine volume'].tail()

# chuyển dung tích động cơ từ object về dạng số
df_data['Engine volume'] = pd.to_numeric(df_data['Engine volume'], errors='coerce')
df_data['Engine volume'].head()

df_data['Doors']

'''
**Sửa lỗi định dang excel trong cột Doors**

**Thay thế tất cả các giá trị ngày tháng theo 
định dạng Excel (ví dụ: "04-May") bằng các danh 
mục thứ bậc đúng (như "2-3", "4-5", ">5").**
'''

df_data['Doors'].unique()

df_data['Doors'] = df_data['Doors'].replace({
    '04-May': '4-5',
    '02-Mar': '2-3',
})
df_data['Doors']

df_data.dtypes

# Loai bo trung lap

df_data.duplicated().sum()
df_data.drop_duplicates(inplace=True)


df_data.drop(columns='ID',inplace=True)  
'''
# Không quan trọng trong quá trình phân tích dữ 
liệu khai phá ( Exploratory Data Analysis) EDA

'''
df_data.dtypes

numeric_cols = df_data.select_dtypes(include=['int64', 'float64']).columns
numeric_cols

'''
Phân tích outliers (giá trị ngoại lệ) trong các cột 
số bằng phương pháp IQR (Interquartile Range)
'''

total_rows = len(df_data) #tổng số dòng trong DataFrame
print(total_rows)

col=1
for col in numeric_cols: #Duyệt qua từng cột số (numeric_cols)
    Q1 = df_data[col].quantile(0.25) # Phân vị thứ 25 (quartile 1)
    Q3 = df_data[col].quantile(0.75) # Phân vị thứ 75 (quartile 3)
    IQR = Q3 - Q1 # Độ rộng khoảng trung vị
    #Xác định ngưỡng để coi là "ngoại lệ"
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_data[(df_data[col] < lower) | (df_data[col] > upper)]
    count = len(outliers)
    percent = (count / total_rows) * 100

    print(f"{col}: {count} rows ({percent:.2f}%)")

'''
:Dù có outliers, nhưng giá trị vẫn có vẻ hợp lý 
( dung tích động cơ cao, số lượng xi-lanh hiếm...), 
nên KHÔNG xóa, vẫn giữ lại trong dữ liệu.
'''

df_data.isna().sum()
#Levy co 5709 gia trj NaN
'''
Thay thế các giá trị NaN (thiếu dữ liệu) 
trong cột Levy bằng giá trị trung vị (median)
'''
df_data['Levy']=df_data['Levy'].fillna(df_data['Levy'].median())
df_data.isna().sum()
# khong con cot nao co NaN nua

# Biểu đồ nhiệt giữa những đặc tính só
plt.figure(figsize=(10,6))
sns.heatmap(df_data[numeric_cols].corr(), annot=True, cmap='Blues')
plt.title('Biểu đồ nhiệt giữa những đặc tính số')
plt.show()

categorical_columns = df_data.select_dtypes(include='object').columns.tolist()
#categorical_cols
print(categorical_columns)

df_data_categorical = df_data.copy()

df_data['Manufacturer'].unique()
df_data['Manufacturer'].nunique()

df_data['Model'].unique()
df_data['Model'].nunique() #1590
for model in df_data['Model'].unique():
    print(model)

# Lưu các giá trị unique vào một mảng
#unique_models = df_data['Model'].unique().tolist()

# Tính số lượng phần tử 1/10 đầu tiên (làm tròn xuống)
#n = len(unique_models) // 10

# In ra 1/10 đầu danh sách
#print(f"In {n} giá trị đầu tiên trong tổng số {len(unique_models)} giá trị khác nhau:\n")
#for i in range(500,950):
#    print(f"{i}. {unique_models[i]}")
    


df_data['Category'].unique()
df_data['Category'].nunique()

df_data['Leather interior'].unique()
df_data['Leather interior'].nunique()

df_data['Fuel type'].unique()
df_data['Fuel type'].nunique()

df_data['Gear box type'].unique()
df_data['Gear box type'].nunique()

df_data['Drive wheels'].unique()
df_data['Drive wheels'].nunique()

df_data['Doors'].unique()
df_data['Doors'].nunique()

df_data['Wheel'].unique()
df_data['Wheel'].nunique()



df_data['Color'].unique()
df_data['Color'].nunique()

#df_data2 = df_data.copy()
#print(df_data2.head())

df_data.drop(columns='Model',inplace=True)
#df_data.drop(columns='Index', inplace = True)  
print(df_data.columns)

y = df_data[['Price']] # bien phu thuoc
#X = df_data.drop('Price', axis=1) # bien doc lap
#df_data2.drop(columns = 'Model')
#print(df_data2)

X = df_data.drop(columns = 'Price' )
X2 = X.copy()
X2= X2.drop(columns='Levy')
X2= X2.drop(columns='Prod. year')
X2= X2.drop(columns='Mileage')
print(X.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 


X_train2 = X_train.copy()
X_test2 = X_test.copy()
y_train2 = y_train.copy()
y_test2 = y_test.copy()


#encoder2 = OrdinalEncoder()

#encoder2.fit(X_train2[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']])

#X_train2[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']] = encoder2.transform(X_train[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']])
                         
#X_test2[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']] = encoder2.transform(X_test[['Manufacturer', 'Model','Category','Leather interior','Fuel type', 'Gear box type','Drive wheels','Doors', 'Wheel', 'Color']])


if 'Model' in categorical_columns:
    categorical_columns.remove('Model')


# 2. Fit encoder trên tập huấn luyện
encoder3 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder3.fit(X_train[categorical_columns])

# 3. Transform cả train và test
X_train[categorical_columns] = encoder3.transform(X_train[categorical_columns])
X_test[categorical_columns] = encoder3.transform(X_test[categorical_columns])


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Danh gia mo hinh
mse = mean_squared_error(y_test, predictions)
r2_square = r2_score(y_test,predictions)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')




