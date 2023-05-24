#pip install streamlit
#pip install jinja2 --upgrade

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import datetime
import warnings
import altair as alt
import plotly.express as px
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def get_similar_top5(vectorizer, tfidf_matrix, user_input):
    # Transform the user input using the fitted vectorizer
    user_input_tfidf = vectorizer.transform(user_input)

    # Compute the cosine similarity between user input and recipe corpus
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)

    # Get the indices of top 5 similar recipes
    similar_top5 = list(cosine_sim.argsort()[0][-5:])

    return similar_top5


def get_names_top5(dataframe, similar_top5):
    # Get the names of top 5 recommended recipes
    recommended_recipes = list(dataframe.iloc[similar_top5].index)
    return recommended_recipes


base_url='/Users/ipekgamzeucal/Desktop/Data_Science/Miuul-DSMLBC11/RecipeBox'
seg1_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg1_vectorizer.pkl','rb'))
seg1_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg1_tfidf_matrix.pkl','rb'))
seg2_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg2_vectorizer.pkl','rb'))
seg2_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg2_tfidf_matrix.pkl','rb'))
seg3_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg3_vectorizer.pkl','rb'))
seg3_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg3_tfidf_matrix.pkl','rb'))
seg4_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg4_vectorizer.pkl','rb'))
seg4_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg4_tfidf_matrix.pkl','rb'))


st.set_page_config(page_title='RecipeBox')
tabs = ['Home Page', 'EDA', 'Forecasting']


page = st.sidebar.radio('Pages', tabs)

current_time = datetime.now().strftime("%d-%m-%Y")

st.sidebar.write('Date: {}'.format(current_time))

#Tekil malzeme listesinin okunmasi
df_final_ingredients=pd.read_csv(f'{base_url}/final_datasets/final_ingredients.csv')
list_final_ingredients=list(df_final_ingredients.INGREDIENT)
ingr_list_for_user = ["Select and add an ingredient"] + list_final_ingredients[:]

#Tekil tag listesinin okunmasi
df_final_tags=pd.read_csv(f'{base_url}/final_datasets/final_tags.csv')
list_final_tags=list(df_final_tags.tag)
tag_list_for_user = ["Select and add a relevant tag"] + list_final_tags[:]

#Tarif ID-Name listesinin okunmasi
final_recipes=pd.read_csv(f'{base_url}/final_datasets/final_repices.csv')
final_recipes_name_indexed=final_recipes.set_index('name')

if page == 'Home Page':

    #Segment seçimi
    #'Less than 40 mins & Low Calorie', 'Less than 40 mins & High Calorie','More than 40 mins & Low Calorie', 'More than 40 mins & High Calorie'
    st.markdown("<h3 style='text-align:center;'>Calorie & Cooking Time</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'>Choose which suits you the best </h3>", unsafe_allow_html=True)
    col1, col2,col3 = st.columns([5,3,5])
    with col1:
        selected_time= st.select_slider(
        'Cooking time',
        options=['Less than 40 mins','More than 40 mins'])
    with col3:
        selected_cal= st.select_slider(
            'Calorie',
            options=['Low Calorie','High Calorie'])


    #User Input
    st.markdown("<h3 style='text-align:center;'>Ingredients & Tags</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'>Select ingredients that you have today, add tags if you want</h3>", unsafe_allow_html=True)

    # Tekil malzemelerin kullanicinin secimi icin listelenmesi
    ing_list = st.multiselect('Select ingredients',ingr_list_for_user)

    # Tekil taglerin kullanicinin secimi icin listelenmesi
    tag_list = st.multiselect('Select tags',tag_list_for_user)

    user_input=ing_list+tag_list

    st.markdown("<h5 style='text-align:center;'>You selected</h3>", unsafe_allow_html=True)
    st.write(selected_time+' & '+selected_cal)
    # Secılen malzeme ve taglerın brlestırılmesı ve ekranda gosterılmesı
    st.write('User Input:', user_input)

    user_model_input=user_input
    # Girilen user input'a göre top 5 tarifin önerilmesi
    similar_top5_indices = get_similar_top5(seg1_vectorizer, seg1_tfidf_matrix, user_model_input)
    recomms=get_names_top5(seg4, similar_top5_indices)
    st.write('Recommendations:', recomms)