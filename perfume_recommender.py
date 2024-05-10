import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the perfume data from GitHub
url = "https://raw.githubusercontent.com/spdini/perfume_recommender/main/Perfume%20Data.csv"
response = requests.get(url)
perfumes = pd.read_csv(StringIO(response.text), encoding='latin1')

# Preprocess the data
perfumes[['Top Notes', 'Heart Notes', 'Based Notes', 'Mood']] = perfumes[['Top Notes', 'Heart Notes', 'Based Notes', 'Mood']].fillna('')
perfumes['tags'] = perfumes[['Top Notes', 'Heart Notes', 'Based Notes', 'Mood']].agg(', '.join, axis=1)
perfumes['tags'] = perfumes['tags'].apply(lambda x: ', '.join(filter(None, x.split(', '))))

# Vectorize the tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(perfumes['tags']).toarray()
similarity = cosine_similarity(vector)

# Filtered dataframe based on selected brand
def filter_by_brand(brand_name):
    return perfumes[perfumes['Brand'] == brand_name]['Product Name'].tolist()

# Recommendation function
def recommend_3(brand_name, perfume):
    index = perfumes[(perfumes['Brand'] == brand_name) & (perfumes['Product Name'] == perfume)].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_perfumes = []
    for i in distances[1:6]:
        recommended_perfume = perfumes.iloc[i[0]]
        recommended_perfumes.append((recommended_perfume['Brand'], recommended_perfume['Product Name']))
    return recommended_perfumes

# Streamlit app
st.title('Perfume Recommender')

# Dropdown list for selecting brand
brand_name = st.selectbox('Select a Brand', perfumes['Brand'].unique())

# Dropdown list for selecting product name based on selected brand
product_names = filter_by_brand(brand_name)
product_name = st.selectbox('Select a Product Name', product_names)

# Button to trigger recommendation
if st.button('Show Similar Products'):
    # Get recommendations
    recommendations = recommend_3(brand_name, product_name)
    
    # Display recommendations
    st.subheader('Similar Perfumes:')
    for i, recommendation in enumerate(recommendations, start=1):
        st.write(f'{i}. {recommendation[0]} / {recommendation[1]}')
