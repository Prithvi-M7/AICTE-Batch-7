import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image
from pathlib import Path


# Load data
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "gld_price_data.csv"
image_path = BASE_DIR / "img.jpeg"

if not data_path.exists():
    st.error(f"Data file not found: {data_path}")
    st.stop()

gold_data = pd.read_csv(data_path)

# Split into X and Y
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']
print(X.shape," \n",Y.shape)
# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=2)
print(X_train.shape,X_test.shape)

reg = RandomForestRegressor()
reg.fit(X_train,Y_train)
pred = reg.predict(X_test)
score = r2_score(Y_test,pred)


# web app
st.title('Gold Price Model')
if image_path.exists():
    img = Image.open(image_path)
    st.image(img, width=200)
else:
    st.warning(f"Image file not found: {image_path}")

st.subheader('Using randomforestregressor')
st.write(gold_data)
st.subheader('Model Performance')
st.write(score)
