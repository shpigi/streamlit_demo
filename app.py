import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI

# Page Configuration
st.set_page_config(
    page_title="Streamlit Demo",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Introduction
def introduction():
    st.title("Welcome to the Streamlit Demo!")
    st.markdown('''
        This app is a demonstration of Streamlit's capabilities, especially for data science and machine learning applications.
        Each tab in the sidebar showcases a different set of features.
    ''')
    st.header("What is Streamlit?")
    st.markdown('''
        Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
        If you know Python, you can create a Streamlit app in a few hours!
    ''')
    st.code('''
# The code for this section
st.title("Welcome to the Streamlit Demo!")
st.markdown("...")
    ''', language='python')

def interactive_widgets():
    st.header("Interactive Widgets & Data Filtering")
    st.markdown("Streamlit's widgets allow you to easily add interactivity to your apps. Here, we're using them to filter the Palmer Penguins dataset.")

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
        return df

    df = load_data()

    # Sidebar widgets for filtering
    st.sidebar.header("Filter Data")
    species = st.sidebar.multiselect("Select Species", df['species'].unique(), df['species'].unique())
    island = st.sidebar.radio("Select Island", df['island'].unique())
    bill_length = st.sidebar.slider("Bill Length (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), (float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max())))

    # Filter data
    filtered_df = df[(df['species'].isin(species)) & (df['island'] == island) & (df['bill_length_mm'] >= bill_length[0]) & (df['bill_length_mm'] <= bill_length[1])]

    st.dataframe(filtered_df)

    st.code('''
# The code for this section
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
    return df

df = load_data()

# Sidebar widgets for filtering
st.sidebar.header("Filter Data")
species = st.sidebar.multiselect("Select Species", df['species'].unique(), df['species'].unique())
island = st.sidebar.radio("Select Island", df['island'].unique())
bill_length = st.sidebar.slider("Bill Length (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), (float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max())))

# Filter data
filtered_df = df[(df['species'].isin(species)) & (df['island'] == island) & (df['bill_length_mm'] >= bill_length[0]) & (df['bill_length_mm'] <= bill_length[1])]

st.dataframe(filtered_df)
''', language='python')

def advanced_charting():
    st.header("Advanced Charting with Plotly")
    st.markdown("Streamlit seamlessly integrates with popular charting libraries like Plotly. This allows for creating interactive and informative visualizations that respond to user input.")

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
        return df

    df = load_data()

    # Sidebar widgets for filtering
    st.sidebar.header("Chart Controls")
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns, key='x_axis')
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns, key='y_axis')

    # Create scatter plot
    fig = px.scatter(df, x=x_axis, y=y_axis, color="species", title=f"{x_axis.replace('_', ' ').title()} vs. {y_axis.replace('_', ' ').title()}")
    st.plotly_chart(fig)

    st.code('''
# The code for this section
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
    return df

df = load_data()

# Sidebar widgets for filtering
st.sidebar.header("Chart Controls")
x_axis = st.sidebar.selectbox("Select X-axis", df.columns, key='x_axis')
y_axis = st.sidebar.selectbox("Select Y-axis", df.columns, key='y_axis')

# Create scatter plot
fig = px.scatter(df, x=x_axis, y=y_axis, color="species", title=f"{x_axis.replace('_', ' ').title()} vs. {y_axis.replace('_', ' ').title()}")
st.plotly_chart(fig)
''', language='python')

def performance_state():
    st.header("Performance & State")
    st.markdown("Streamlit provides tools to optimize performance and manage state.")

    # Caching
    st.subheader("Caching")
    st.markdown("`st.cache_data` is used to cache the output of functions. This is useful for expensive operations like loading data from a database or performing complex calculations. Notice how the data loading is instant after the first time.")
    st.code('''
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
    return df
''', language='python')

    # Session State
    st.subheader("Session State")
    st.markdown("`st.session_state` allows you to preserve information across user interactions and reruns. Here, we're using it to count the number of times a button is clicked.")
    
    if 'click_count' not in st.session_state:
        st.session_state.click_count = 0

    if st.button("Click me"):
        st.session_state.click_count += 1

    st.write(f"Button clicked {st.session_state.click_count} times.")
    st.code('''
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0

if st.button("Click me"):
    st.session_state.click_count += 1

st.write(f"Button clicked {st.session_state.click_count} times.")
''', language='python')

def ml_integration():
    st.header("Machine Learning Integration")
    st.markdown("Streamlit makes it easy to build interactive machine learning applications. Here, we're using a pre-trained Random Forest model to predict penguin species.")

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
        df = df.dropna()
        return df

    df = load_data()

    # Train model
    @st.cache_resource
    def train_model():
        X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
        y = df['species']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, X.columns

    model, feature_names = train_model()

    # User input
    with st.form("prediction_form"):
        st.subheader("Predict Penguin Species")
        bill_length_mm = st.number_input("Bill Length (mm)", min_value=0.0, value=40.0)
        bill_depth_mm = st.number_input("Bill Depth (mm)", min_value=0.0, value=18.0)
        flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, value=200.0)
        body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, value=4000.0)
        island = st.selectbox("Island", df['island'].unique())
        sex = st.selectbox("Sex", df['sex'].unique())
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame({
            'bill_length_mm': [bill_length_mm],
            'bill_depth_mm': [bill_depth_mm],
            'flipper_length_mm': [flipper_length_mm],
            'body_mass_g': [body_mass_g],
            'island_Torgersen': [1 if island == 'Torgersen' else 0],
            'island_Dream': [1 if island == 'Dream' else 0],
            'sex_male': [1 if sex == 'male' else 0]
        })
        
        # Ensure columns match model's expected columns
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Species: **{prediction[0]}**")

    st.code('''
# The code for this section
# ... (code for loading data, training model, and user input form) ...
''', language='python')

def llm_conversation():
    st.header("Conversing with OpenAI's gpt-4o")
    st.markdown("This tab demonstrates how to build a conversational AI with Streamlit and OpenAI's `gpt-4o` model.")

    # Get API key
    api_key = st.text_input("Enter your OpenAI API key", type="password")

    if not api_key:
        st.info("Please enter your OpenAI API key to proceed.")
        st.markdown("You can get a key from [OpenAI](https://platform.openai.com/account/api-keys).")
        return

    client = OpenAI(api_key=api_key)

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages,
                stream=True,
            )
            for chunk in stream:
                full_response += chunk.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.code('''
# The code for this section
# ... (code for getting API key, initializing chat, and handling chat input) ...
''', language='python')


# Main App
def main():
    st.sidebar.title("Navigation")
    tabs = {
        "Introduction": introduction,
        "Interactive Widgets & Data Filtering": interactive_widgets,
        "Advanced Charting": advanced_charting,
        "Performance & State": performance_state,
        "ML Integration": ml_integration,
        "LLM Conversation": llm_conversation
    }
    selection = st.sidebar.radio("Go to", list(tabs.keys()))

    # Placeholder for other tabs
    if tabs[selection]:
        tabs[selection]()
    else:
        st.write("This tab is under construction.")

if __name__ == "__main__":
    main()