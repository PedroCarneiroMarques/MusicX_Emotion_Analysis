import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

################################################ BACKGROUND & STYLE ########################################################

# Add a background image to Streamlit
def set_background():
    background_image = 'https://i.imgur.com/tPHIgmR.png'
    background_html = f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_html, unsafe_allow_html=True)

# Run the function to set the background image
set_background()

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a page",
    ["Home", "Data Visualizations", "Machine Learning Model", "Documentation"]
)

################################################ HOME ########################################################

# Load your dataset here
reviews = pd.read_csv('music_survey.csv')

if page == "Home":
    # Adding welcome title using markdown
    st.markdown("<h1 style='text-align: center;'>Welcome to MusicX Mental Health</h1>", unsafe_allow_html=True)

    # Adding space between the title and content
    st.markdown("<br>", unsafe_allow_html=True)

    # Adding the main content using markdown
    st.markdown(
        """
        <p style="text-align: justify;">
        Music therapy (MT) uses music to enhance mental health. The MxMH dataset explores correlations between music 
        preferences and self-reported mental health. </p>
        
        <p style="text-align: justify;">
        The Respondents ranked how often they listen to 16 music genres, where they can select: Never, Rarely, Sometimes, Very frequently </p>
        
        <p style="text-align: justify;">
        The Respondents ranked Anxiety, Depression, Insomnia, and OCD on a scale of 0 to 10, where: 0 - I do not experience this. 10 - I experience this regularly, constantly/or to an extreme </p>
        
        <p style="text-align: justify;">
        This Data aims to offer insights into these relationships. The collection process involved a Google Form distributed across online forums and physical locations.
        </p>
        
        <p style="text-align: justify;">
        Information was collected between: 27th of August 2022 and 9th of November 2022 with a total of 737 entries.
        </p>
        """, unsafe_allow_html=True
    )

################################################ VISUALIZATIONS ########################################################

elif page == "Data Visualizations":
    st.markdown("<h1 style='text-align: center;'>Data Visualizations</h1>", unsafe_allow_html=True)
    st.sidebar.title("Choose an option")

    data_vis_option = st.sidebar.selectbox(
        "Choose an option",
        ["Age Distribution", "Hours Distribution", "Streaming Service Count", "Mental Health Symptoms", "Responses while Working", "Favorite Genre Distribution", "New Music", "Genre Preferences vs. Age", "Correlation Heatmap", "Music Effect Pie Chart"]
    )

    if data_vis_option == "Age Distribution":
        fig_age = px.histogram(reviews, x='Age', nbins=20, title='Distribution of Age')
        fig_age.update_layout(
            xaxis=dict(title='Age'),
            yaxis=dict(title='Frequency'),
            title='Distribution of Age with Bins Slider'
        )
        st.plotly_chart(fig_age)

    elif data_vis_option == "Hours Distribution":
        num_bins = st.slider("Select the number of bins:", min_value=5, max_value=30, value=15)
        fig_hours = plt.figure(figsize=(8, 6))
        plt.hist(reviews['Hours per day'], bins=num_bins, color='salmon', edgecolor='black')
        plt.xlabel('Hours per day')
        plt.ylabel('Frequency')
        plt.title('Distribution of Hours per day')
        st.pyplot(fig_hours)

    elif data_vis_option == "Streaming Service Count":
        service_counts = reviews['Primary streaming service'].value_counts().reset_index()
        service_counts.columns = ['Service', 'Count']
        fig_service = px.bar(service_counts, x='Service', y='Count', color='Service', title='Primary Streaming Service Counts')
        st.plotly_chart(fig_service)

    elif data_vis_option == "Mental Health Symptoms":
        fig_mental = px.box(reviews, y=['Anxiety', 'Depression', 'Insomnia', 'OCD'], title='Boxplot of Mental Health Symptoms')
        st.plotly_chart(fig_mental)

    elif data_vis_option == "Responses while Working":
        st.subheader("Count of Responses while Working")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        sns.countplot(x='While working', data=reviews[reviews['While working'] != 0], palette='Set2', ax=ax1)
        st.pyplot(fig1)

    elif data_vis_option == "Favorite Genre Distribution":
        st.subheader("Distribution of Favorite Genres")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.countplot(y='Fav genre', data=reviews[reviews['Fav genre'] != 0], palette='viridis', order=reviews['Fav genre'].value_counts().index, ax=ax2)
        st.pyplot(fig2)

    elif data_vis_option == "New Music":
        st.subheader("Users Exploring New Music")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        sns.countplot(x='Exploratory', data=reviews[reviews['Exploratory'] != 0], palette='viridis', ax=ax3)
        st.pyplot(fig3)

    elif data_vis_option == "Genre Preferences vs. Age":
        st.subheader("Genre Preferences vs. Age")
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        violin = sns.violinplot(x='Fav genre', y='Age', data=reviews, palette='viridis', ax=ax6)
        violin.set_xticklabels(violin.get_xticklabels(), rotation=70)  # Rotating x-axis labels
        plt.xlabel('Favorite Genre')
        plt.ylabel('Age')
        plt.title('Genre Preferences Across Age Groups')
        st.pyplot(fig6)

    elif data_vis_option == "Correlation Heatmap":
        numeric_cols = reviews.select_dtypes(include=['float64', 'int64'])
        st.subheader("Correlation Heatmap")
        fig7, ax7 = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax7)
        plt.title('Correlation Heatmap of Numeric Features')
        st.pyplot(fig7)

    elif data_vis_option == "Music Effect Pie Chart":
        st.subheader("Music Effect Distribution")
        music_effect_counts = reviews['Music effects'].value_counts().reset_index()
        music_effect_counts.columns = ['Music Effect', 'Count']
        fig10, ax10 = plt.subplots(figsize=(6, 6))
        ax10.pie(music_effect_counts['Count'], labels=music_effect_counts['Music Effect'], autopct='%1.1f%%', colors=sns.color_palette('Set3'), startangle=140)
        plt.axis('equal')
        plt.title('Music Effect Distribution')
        st.pyplot(fig10)
    
################################################ MACHINE LEARNING ########################################################

elif page == "Machine Learning Model":
    st.markdown("<h1 style='text-align: center;'>Machine Learning Model</h1>", unsafe_allow_html=True)
    st.write("This page contains the machine learning model content.")
    # Add your machine learning model code and interactive elements here
    # For example:
    # Load your dataset here
    reviews = pd.read_csv('music_survey.csv')
    
    # List of columns containing frequency information
    frequency_columns = ['Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]']

    # Replace categorical values with numeric values for frequency columns
    # Assign numerical values to the categories (e.g., 'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3)
    mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
    reviews[frequency_columns] = reviews[frequency_columns].replace(mapping)

    # Create a new feature 'TotalMusicFreq' by summing up frequencies of different music genres
    reviews['TotalMusicFreq'] = reviews[frequency_columns].sum(axis=1)

    # Define features (X) and the target variable (y)
    emotional_features = ['Anxiety', 'Depression', 'Insomnia','OCD']
    music_features = ['Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]']
    X = reviews[emotional_features + music_features]
    y = reviews['TotalMusicFreq']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model with the best hyperparameters
    random_forest = RandomForestRegressor(n_estimators=500, random_state=42)

    # Train the model
    random_forest.fit(X_train, y_train)

    # Make predictions
    predictions = random_forest.predict(X_test)

    # Evaluate model performance using mean squared error
    mse = mean_squared_error(y_test, predictions)
    
    # Display model evaluation metrics or predictions
    st.write("Mean Squared Error:", mse)    

    # Sidebar for feature selection and sample size
    st.sidebar.title('Feature Selection & Sample Size')

    # Feature selection
    selected_features = st.sidebar.multiselect('Select Features', emotional_features + music_features)
    st.sidebar.write('Selected Features:', selected_features)

    # Sample size slider
    sample_size = st.sidebar.slider('Sample Size', min_value=1, max_value=736, value=10)

    # Display selected features
    st.write('Selected Features:', selected_features)

    # Split the data
    train_data, test_data = train_test_split(reviews, test_size=0.2, random_state=42)

    # Train the model and generate predictions only if features are selected from the predefined list
    if selected_features and set(selected_features).issubset(set(emotional_features + music_features)):
        # Train the model on the entire dataset
        X = train_data[selected_features]
        y = train_data['TotalMusicFreq']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict based on user-selected sample size
        predictions = model.predict(test_data[selected_features][:sample_size])

        # Evaluate model performance using mean squared error
        mse = mean_squared_error(test_data['TotalMusicFreq'][:sample_size], predictions)

        # Display predictions and evaluation metrics
        st.write('Predictions:')
        st.write(predictions)

        st.write(f"Mean Squared Error: {mse}")
    else:
        st.write("Please select features from the predefined list to train the model.")

################################################ DOCUMENTATION ########################################################

elif page == "Documentation":
    st.markdown("<h1 style='text-align: center;'>Documentation</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>MusicX Mental Health Project</h2>", unsafe_allow_html=True)
    st.write("Overview: This documentation aims to provide insights into the correlation between music preferences and self-reported mental health. Collected data spans responses from 737 individuals regarding music genre preferences and self-reported mental health issues.")

    st.markdown("<h3 style='text-align: center;'>Data Overview</h3>", unsafe_allow_html=True)
    st.write("Description: The dataset comprises self-reported responses on music genre preferences and mental health symptoms. It includes 737 entries and features columns such as music genre frequencies, mental health ratings for Anxiety, Depression, Insomnia, and OCD, and demographics like Age and Hours per day listening to music.")

    st.markdown("<h3 style='text-align: center;'>Methodology</h3>", unsafe_allow_html=True)
    st.write("Process: Initial steps included preprocessing by converting categorical frequency values into numeric representations. Feature engineering was performed by creating a 'TotalMusicFreq' feature from frequency columns. The dataset was split into training and testing sets for machine learning model creation")
    
    st.markdown("<h3 style='text-align: center;'>Libraries Used</h3>", unsafe_allow_html=True)
    st.write("Listing: Primary tools used involve Pandas for data manipulation, Scikit-learn for machine learning tasks, Streamlit for the app interface, and Plotly for data visualization purposes.")

    st.markdown("<h3 style='text-align: center;'>Findings and Conclusion</h3>", unsafe_allow_html=True)
    st.write("Insights: Initial analysis reveals potential associations between certain music genre frequencies and reported mental health symptoms. Further investigation is needed to establish causal relationships.")

    st.markdown("<h3 style='text-align: center;'>Future Work</h3>", unsafe_allow_html=True)
    st.write("Prospects: Future directions involve deeper analysis using more sophisticated models or conducting surveys to gather more diverse data. Incorporating qualitative data or conducting longitudinal studies could provide richer insights.")
