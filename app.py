import streamlit as st
from PIL import Image
import pandas as pd
from bert_search import SingletonSearch
from okapi_bm25_search import OkapiBM25Search

bert_search = SingletonSearch()
okapi_search = OkapiBM25Search()

# Load the CSV file into a DataFrame
df = pd.read_csv("datasets/model_rating.csv")

# update the number of satisfaction of each model
def update_model_satisfaction(df, model_name,type):
    index = df.index[df["model_name"] == model_name].tolist()[0]
    if type == 'increase':
        df.at[index, "num_of_rating"] += 1
    else:
        df.at[index, "num_of_rating"] -= 1

    # Save the updated DataFrame back to the CSV file
    df.to_csv("datasets/model_rating.csv", index=False)


def display_movies(data):
    if data is None:
        st.info("No News")
    else:
        for i in range(len(data)):
            story = data.iloc[i]
            title = story["title"]
            image = Image.open("assets/netflix.png")
        
            if title is not None:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if image is not None: st.image(image,width=150)
                with col2:
                    st.markdown('<style>.title {font-weight: bold;font-size:20px}</style>', unsafe_allow_html=True)
                    st.markdown(f'<p class="title">{title}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p >{story["description"]}</p>', unsafe_allow_html=True)
                    st.write(f"Durations: {story['duration']}, Realease Year: {story['release_year']}&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;Relevant Score: {round(story['score'],5)}")
def sidebar():
    with st.sidebar:
        st.markdown('<style>.rate {font-weight: bold;font-size:20px;color:red}</style>', unsafe_allow_html=True)
        st.markdown(f'<p>Model : <span class="rate">BERT</span> => Number of Satisfaction:<span class="rate"> {df.iloc[0]["num_of_rating"]}</span></p>', unsafe_allow_html=True)
        st.markdown(f'<p>Model : <span class="rate">Okapi BM25</span> => Number of Satisfaction:<span class="rate"> {df.iloc[1]["num_of_rating"]}</span></p>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Netflix Movie Search Engine",
        page_icon="🎥",
        initial_sidebar_state="expanded",
        menu_items={"About": "Built by @dcarpintero with Streamlit & NewsAPI"},
    )

    st.title(">> Netflix Movies Search Engine <<")
    # Add input elements
    # Create a text input for the user to enter the search query
    search_query = st.text_input("Enter the Movie's Title:")

    option = st.radio(
        label="Choose a Model:",
        options=("BERT", "Okapi BM25")
    )

    data, duration = bert_search.search(search_query) if option == "BERT" else okapi_search.search(search_query)

    if not duration == 0:
        st.markdown('<style>.duration {font-weight: bold}</style>', unsafe_allow_html=True)
        st.markdown(f'<p>Results after <span class="duration">{round((duration/1000),3)}</span> seconds</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4,1,1])

    with col1:
        st.markdown(f"Do you satisfy with the results using this Model ({option})?")
    with col2:
        # Display the button
        like_button = st.button("Like 👍")
        if like_button:
            if option == "BERT":
                update_model_satisfaction(df,"bert","increase")
            else:
                update_model_satisfaction(df,"okapi","increase")            
    with col3:
        # Display the button
        dislike_button = st.button("Dislike 👍")
        if dislike_button:
            if option == "BERT":
                update_model_satisfaction(df,"bert","decrease")
            else:
                update_model_satisfaction(df,"okapi","decrease")
            
    sidebar()
    display_movies(data)

if __name__ == "__main__":
    main()


