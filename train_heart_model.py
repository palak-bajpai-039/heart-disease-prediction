import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('heart.csv')

df['Cholesterol'] = df['Cholesterol'].replace(
    0,
    df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean(),
)
df['RestingBP'] = df['RestingBP'].replace(
    0,
    df.loc[df['RestingBP'] != 0, 'RestingBP'].mean(),
)

encoded = pd.get_dummies(df, drop_first=True)
encoded = encoded.astype(int)

x = encoded.drop(columns=['HeartDisease'])
y = encoded['HeartDisease']

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier()
model.fit(x_train, y_train)

joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x.columns.tolist(), 'columns.pkl')

print('Model retrained and saved successfully.')
print('Feature count:', len(x.columns))
