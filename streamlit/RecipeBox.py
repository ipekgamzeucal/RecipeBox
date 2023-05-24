import pickle
import ast
import streamlit_extras
from streamlit_extras.switch_page_button import switch_page
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
import streamlit as st
from PIL import Image
from io import BytesIO
import requests

def main():
    response = requests.get('https://raw.githubusercontent.com/ipekgamzeucal/RecipeBox/main/assets/simgemiz.png')
    image = Image.open(BytesIO(response.content))
    st.image(image, caption='', use_column_width=False)
    st.markdown(
        """
        <style>
        .main {
            background-color: black;
            backdrop-filter: blur(10px);
        }

        header{
            background: black !Important;
        }
        ul li {
            color: white;
        }
        .stMarkdown{
        color : white;
        }
        .css-1vbkxwb p{
        color: rgb(255, 75, 75) !Important;
        }
        .css-1y4p8pa{
            padding: 3rem 1rem 10rem !Important;
        }
        .css-1kyxreq{
            JUSTIFY-CONTENT: center !IMPORTANT;
        }
        .css-vk3wp9{
        background-color: rgb(226 157 157) !Important;
        }
        
        .css-fblp2m{
        color : white !Important;
        }
        p {
            color: white;
        }
        h3,h2,h5,h4{
            color: white;
        }
        .e88czh80{
        color: white
        }
               
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Recipe Box", page_icon="\U0001F60B",layout="wide",
    initial_sidebar_state="expanded")
    main()



def tf_idf_vectorizer(dataframe, corpus_col):
        # Create the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")

        # Fit and transform the recipe corpus
        recipe_corpus = dataframe[corpus_col]
        tfidf_matrix = vectorizer.fit_transform(recipe_corpus)

        return vectorizer, tfidf_matrix

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



#base_url='https://raw.githubusercontent.com/ipekgamzeucal/RecipeBox/main'
#seg1_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg1_vectorizer.pkl','rb'))
#seg1_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg1_tfidf_matrix.pkl','rb'))
#seg2_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg2_vectorizer.pkl','rb'))
#seg2_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg2_tfidf_matrix.pkl','rb'))
#seg3_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg3_vectorizer.pkl','rb'))
#seg3_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg3_tfidf_matrix.pkl','rb'))
#seg4_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg4_vectorizer.pkl','rb'))
#seg4_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg4_tfidf_matrix.pkl','rb'))

base_url='https://raw.githubusercontent.com/ipekgamzeucal/RecipeBox/main'
seg1_vectorizer = pd.read_pickle(f'{base_url}/content_based_pickles/seg1_vectorizer.pkl')
seg1_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg1_tfidf_matrix.pkl','rb'))
seg2_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg2_vectorizer.pkl','rb'))
seg2_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg2_tfidf_matrix.pkl','rb'))
seg3_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg3_vectorizer.pkl','rb'))
seg3_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg3_tfidf_matrix.pkl','rb'))
seg4_vectorizer = pickle.load(open(f'{base_url}/content_based_pickles/seg4_vectorizer.pkl','rb'))
seg4_tfidf_matrix = pickle.load(open(f'{base_url}/content_based_pickles/seg4_tfidf_matrix.pkl','rb'))

filtered_recipes=pd.read_csv(f'{base_url}/final_datasets/final_repices_all.csv')
seg1=filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories<=400)]
seg2=filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories>400)]
seg3=filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories<=400)]
seg4=filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories>400)]
seg1.set_index('name', inplace=True)
seg2.set_index('name', inplace=True)
seg3.set_index('name', inplace=True)
seg4.set_index('name', inplace=True)


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

## ITEM_BASED listesinin okunması

item_based_recom = pd.read_csv(f'{base_url}/ItembasedRecommender/item_based_updated.csv')


def home_page():
    st.sidebar.title("RecipeBox")
    page = st.sidebar.radio("",
                                ("Add Your Ingredients", "RecipeBox"))
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.info(' İpek Gamze UÇAL ', icon="🐨")
    st.sidebar.info(' Işın KOÇER', icon="🐼")
    st.sidebar.info(' Büşra ŞENTÜRK ', icon="🐈")
    st.sidebar.info(' Şeyma NURDAĞ ', icon="🌼")

    if page == 'Add Your Ingredients':
        # page1()
        #Segment seçimi
        #'Less than 40 mins & Low Calorie', 'Less than 40 mins & High Calorie','More than 40 mins & Low Calorie', 'More than 40 mins & High Calorie'
        st.write("")
        st.write("")
        st.write("")
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
        st.markdown("<h3 style='text-align:center;'>Select ingredients that you have today, add tags if you want</h3>", unsafe_allow_html=True)

        # Tekil malzemelerin kullanicinin secimi icin listelenmesi
        st.write("Select ingredients")
        ing_list = st.multiselect('Select ingredients',ingr_list_for_user,label_visibility= "collapsed")

        # Tekil taglerin kullanicinin secimi icin listelenmesi
        st.write("Select tags")
        tag_list = st.multiselect('Select tags',tag_list_for_user,label_visibility= "collapsed")

        user_input=ing_list+tag_list

        # st.markdown("<h5 style='text-align:center;'>You selected</h3>", unsafe_allow_html=True)
        # st.write(selected_time+' & '+selected_cal)
        # Secılen malzeme ve taglerın brlestırılmesı ve ekranda gosterılmesı
        # st.write('User Input:', user_input)

        user_model_input = user_input

        # Girilen user input'a göre top 5 tarifin önerilmesi

        if user_input==[]:

            st.markdown("<h5 style='text-align:center;'>You havent selected!</h3>", unsafe_allow_html=True)
            # st.write('You havent selected!')

        else:
            st.markdown("<h5 style='text-align:center;'>You selected</h3>", unsafe_allow_html=True)
            st.write(selected_time + ' & ' + selected_cal)
            # st.write(f"<div style='width: 700px'>{list(user_model_input)}</div>", unsafe_allow_html=True)

            if selected_cal == 'Low Calorie' and selected_time == 'Less than 40 mins':

                similar_top5_indices = get_similar_top5(seg1_vectorizer, seg1_tfidf_matrix, user_model_input)
                recomms = get_names_top5(seg1, similar_top5_indices)
                # seg1_vectorizer, seg1_tfidf_matrix = tf_idf_vectorizer(seg1, 'merged_tags_ingredients')
                # similar_top5_indices = get_similar_top5(seg1_vectorizer, seg1_tfidf_matrix, user_model_input)
                # recomms = get_names_top5(seg1, similar_top5_indices)
                # st.write('Recommendations:', recomms)
                st.write("")
                st.write("")
                st.write("")
                st.write("")

                st.markdown("<h3 style='text-align:center;'>Most Suitable Recipes For You</h3>", unsafe_allow_html=True)
                for item in recomms:

                    if st.button(item.title()):
                        st.write("Let's get recipe from the RecipeBox")
                        # Kullanıcı düğmeye tıkladığında yapılacak işlemler
                        # st.write("Seçilen öğe kalorisi:",f"{list(filtered_recipes[filtered_recipes.name==item].calories)}")
                        # st.markdown(f"- {list(filtered_recipes[filtered_recipes.name==item].calories)}" )
                        st.session_state['recom'] = item

            elif selected_cal == 'High Calorie' and selected_time == 'Less than 40 mins':

                similar_top5_indices = get_similar_top5(seg2_vectorizer, seg2_tfidf_matrix, user_model_input)
                recomms = get_names_top5(seg2, similar_top5_indices)
                # st.write('Recommendations:', recomms)
                st.markdown("<h3 style='text-align:center;'>Most Suitable Recipes For You</h3>", unsafe_allow_html=True)
                for item in recomms:

                    if st.button(item.title()):
                        st.write("Let's get recipe from the RecipeBox")
                        # Kullanıcı düğmeye tıkladığında yapılacak işlemler
                        # st.write("Seçilen öğe kalorisi:",f"{list(filtered_recipes[filtered_recipes.name==item].calories)}")
                        st.session_state['recom'] = item

            elif selected_cal == 'Low Calorie' and selected_time == 'More than 40 mins':
                # seg3_vectorizer, seg3_tfidf_matrix = tf_idf_vectorizer(seg3, 'merged_tags_ingredients')
                similar_top5_indices = get_similar_top5(seg3_vectorizer, seg3_tfidf_matrix, user_model_input)
                recomms = get_names_top5(seg3, similar_top5_indices)
                # st.write('Recommendations:', recomms)
                st.markdown("<h3 style='text-align:center;'>Most Suitable Recipes For You</h5>", unsafe_allow_html=True)
                for item in recomms:
                    if st.button(item.title()):
                        st.write("Let's get recipe from the RecipeBox")
                        # Kullanıcı düğmeye tıkladığında yapılacak işlemler
                        # st.write("Seçilen öğe kalorisi:",f"{list(filtered_recipes[filtered_recipes.name==item].calories)}")
                        st.session_state['recom'] = item
            elif selected_cal == 'High Calorie' and selected_time == 'More than 40 mins':
                # seg4_vectorizer, seg4_tfidf_matrix = tf_idf_vectorizer(seg4, 'merged_tags_ingredients')
                similar_top5_indices = get_similar_top5(seg4_vectorizer, seg4_tfidf_matrix, user_model_input)
                recomms = get_names_top5(seg4, similar_top5_indices)
                # st.write('Recommendations:', recomms)
                st.markdown("<h3 style='text-align:center;'>Most Suitable Recipes For You</h3>", unsafe_allow_html=True)
                for item in recomms:
                    if st.button(item.title()):
                        st.write("Let's get recipe from the RecipeBox")
                        # Kullanıcı düğmeye tıkladığında yapılacak işlemler
                        st.write("")
                        st.session_state['recom'] = item


    if page == 'RecipeBox':
        # page2()
        # st.markdown("<h3 style='text-align:center;'>Recommendations</h3>", unsafe_allow_html=True)
        # st.write(st.session_state.recom)
        id_recom = final_recipes[final_recipes.name == st.session_state.recom].id.iloc[0]
        # st.write(id_recom)

        #########################################
        description = filtered_recipes[filtered_recipes.name == st.session_state.recom]['description'].values[0].capitalize()
        ingredients_list = filtered_recipes.loc[filtered_recipes.name == st.session_state.recom, 'ingredients']
        separated_list = [item for ingredient in ingredients_list for item in ingredient.split(",")]
        formatted_list = [item.replace("'", "").replace("]", "").replace("[", "").replace(",", "") for item in
                          separated_list]

        step_list = filtered_recipes.loc[filtered_recipes.name == st.session_state.recom, 'steps']
        formatted_step_list = [item.replace("'", "").replace("]", "").replace("[", "") for item in step_list]
        recipe_name = st.session_state.recom.title()
        # Segment seçimi
        # 'Less than 40 mins & Low Calorie', 'Less than 40 mins & High Calorie','More than 40 mins & Low Calorie', 'More than 40 mins & High Calorie'
        st.markdown(f"<h3 style='text-align:center;'>{recipe_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align:center;'>{description}</h3>", unsafe_allow_html=True)
        st.write("<span style='font-weight:bold;'>Ingredients</span>", unsafe_allow_html=True)
        # st.write('Ingredients')
        st.markdown("\n".join(f"-  {item}" for item in formatted_list))
        st.write("<span style='font-weight:bold;'>Recipe Steps</span>", unsafe_allow_html=True)
        st.markdown("\n".join(f"-  {item}" for item in formatted_step_list))
        #########################################
        #seçilen yemeğin tarifi

        # st.markdown("<h5 style='text-align:center;'>RECİPE</h3>", unsafe_allow_html=True)
        # id_steps =list(filtered_recipes[filtered_recipes.name == st.session_state.recom].steps)
        # st.write(f"<div style='width: 700px'>{id_steps}</div>", unsafe_allow_html=True)

        ## boşluk koymak için
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        # df2 = pd.DataFrame(ast.literal_eval(item_based_recom[item_based_recom.recipe_id == id_recom].recom_names.iloc[0]))
        # df3 = pd.DataFrame(ast.literal_eval(item_based_recom[item_based_recom.recipe_id == id_recom].recom_ids.iloc[0]))
        # df4 = pd.concat([df2, df3], axis=1)
        # df4.columns = ['recom_names', 'recom_ids']

        # burada item_based için düzenlemeler yapıldı.

        filtered_recom = item_based_recom[item_based_recom.recipe_id == id_recom]

        # Öneri hiç yoksa bi yazı yazdırabiliriz. Yazıya karar verelim
        if filtered_recom.recipe_dict.isnull().any():
            print("Those who like this recipe haven't liked other recipes yet :(")
        else:  # Öneri varsa idlerini ve isimlerini recom_df adında bir dataframe de tutar.
            recipe_dict = ast.literal_eval(filtered_recom.recipe_dict.iloc[0])
            df4 = pd.DataFrame(list(recipe_dict.items()), columns=['recom_ids','recom_names'])

         # recom_df kullanılarak item based öneriler ekranda gösterilebilir

        if item_based_recom[item_based_recom.recipe_id == id_recom].recom_names.isnull().any():
            st.write("Those who like this recipe haven't liked other recipes yet :(")
        else:
            st.markdown("<h3 style='text-align:center;'>Most Liked Recipes by RecipeBox Users !</h3>",
                        unsafe_allow_html=True)
            for item in df4.recom_names:

                if st.button(item.title()):
                    # st.write("Seçilen öğe kalorisi:",
                    #          f"{list(filtered_recipes[filtered_recipes.name == item].calories)}")
                    # Kullanıcı düğmeye tıkladığında yapılacak işlemler
                    description = filtered_recipes[filtered_recipes.name == item]['description'].values[0].capitalize()
                    ingredients_list = filtered_recipes.loc[filtered_recipes.name == item, 'ingredients']
                    separated_list = [item for ingredient in ingredients_list for item in ingredient.split(",")]
                    formatted_list = [item.replace("'", "").replace("]", "").replace("[", "").replace(",", "") for item in separated_list]
                    step_list = filtered_recipes.loc[filtered_recipes.name == item, 'steps']
                    formatted_step_list = [item.replace("'", "").replace("]", "").replace("[", "") for item in step_list]
                    recipe_name = item.title()
                    st.markdown(f"<h3 style='text-align:center;'>{recipe_name}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='text-align:center;'>{description}</h3>", unsafe_allow_html=True)
                    st.write("<span style='font-weight:bold;'>Ingredients</span>", unsafe_allow_html=True)
                    st.markdown("\n".join(f"- {item}" for item in formatted_list))
                    st.write("<span style='font-weight:bold;'>Recipe Steps</span>", unsafe_allow_html=True)

                    # for item in formatted_step_list:
                    #     st.markdown(f"<h3 style='text-align:center;'>{item}</h3>", unsafe_allow_html=True)

                    st.markdown("\n".join(f"- {item}" for item in formatted_step_list))

home_page()
