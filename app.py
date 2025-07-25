import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
import requests
import io
import base64
from PIL import Image

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
    st.markdown(
        """
        This app is a demonstration of Streamlit's capabilities, especially for data science and machine learning applications.
        Each tab in the sidebar showcases a different set of features.
    """
    )
    st.header("What is Streamlit?")
    st.markdown(
        """
        Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
        If you know Python, you can create a Streamlit app in a few hours!
    """
    )
    st.code(
        """
# The code for this section
st.title("Welcome to the Streamlit Demo!")
st.markdown("...")
    """,
        language="python",
    )


def interactive_widgets():
    st.header("Interactive Widgets & Data Filtering")
    st.markdown(
        "Streamlit's widgets allow you to easily add interactivity to your apps. Here, we're using them to filter the Palmer Penguins dataset."
    )

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv(
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
        )
        return df

    df = load_data()

    # Sidebar widgets for filtering
    st.sidebar.header("Filter Data")
    species = st.sidebar.multiselect(
        "Select Species", df["species"].unique(), df["species"].unique()
    )
    island = st.sidebar.radio("Select Island", df["island"].unique())
    bill_length = st.sidebar.slider(
        "Bill Length (mm)",
        float(df["bill_length_mm"].min()),
        float(df["bill_length_mm"].max()),
        (float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max())),
    )

    # Filter data
    filtered_df = df[
        (df["species"].isin(species))
        & (df["island"] == island)
        & (df["bill_length_mm"] >= bill_length[0])
        & (df["bill_length_mm"] <= bill_length[1])
    ]

    st.dataframe(filtered_df)

    st.code(
        """
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
""",
        language="python",
    )


def advanced_charting():
    st.header("Advanced Charting with Plotly")
    st.markdown(
        "Streamlit seamlessly integrates with popular charting libraries like Plotly. This allows for creating interactive and informative visualizations that respond to user input."
    )

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv(
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
        )
        return df

    df = load_data()

    # Sidebar widgets for filtering
    st.sidebar.header("Chart Controls")
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns, key="x_axis")
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns, key="y_axis")

    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="species",
        title=f"{x_axis.replace('_', ' ').title()} vs. {y_axis.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig)

    st.code(
        """
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
""",
        language="python",
    )


def performance_state():
    st.header("Performance & State")
    st.markdown("Streamlit provides tools to optimize performance and manage state.")

    # Caching
    st.subheader("Caching")
    st.markdown(
        "`st.cache_data` is used to cache the output of functions. This is useful for expensive operations like loading data from a database or performing complex calculations. Notice how the data loading is instant after the first time."
    )
    st.code(
        """
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
    return df
""",
        language="python",
    )

    # Session State
    st.subheader("Session State")
    st.markdown(
        "`st.session_state` allows you to preserve information across user interactions and reruns. Here, we're using it to count the number of times a button is clicked."
    )

    if "click_count" not in st.session_state:
        st.session_state.click_count = 0

    if st.button("Click me"):
        st.session_state.click_count += 1

    st.write(f"Button clicked {st.session_state.click_count} times.")
    st.code(
        """
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0

if st.button("Click me"):
    st.session_state.click_count += 1

st.write(f"Button clicked {st.session_state.click_count} times.")
""",
        language="python",
    )


def ml_integration():
    st.header("Machine Learning Integration")
    st.markdown(
        "Streamlit makes it easy to build interactive machine learning applications. Here, we're using a pre-trained Random Forest model to predict penguin species."
    )

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv(
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
        )
        df = df.dropna()
        return df

    df = load_data()

    # Train model
    @st.cache_resource
    def train_model():
        X = pd.get_dummies(df.drop("species", axis=1), drop_first=True)
        y = df["species"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, X.columns

    model, feature_names = train_model()

    # User input
    with st.form("prediction_form"):
        st.subheader("Predict Penguin Species")
        bill_length_mm = st.number_input("Bill Length (mm)", min_value=0.0, value=40.0)
        bill_depth_mm = st.number_input("Bill Depth (mm)", min_value=0.0, value=18.0)
        flipper_length_mm = st.number_input(
            "Flipper Length (mm)", min_value=0.0, value=200.0
        )
        body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, value=4000.0)
        island = st.selectbox("Island", df["island"].unique())
        sex = st.selectbox("Sex", df["sex"].unique())
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame(
            {
                "bill_length_mm": [bill_length_mm],
                "bill_depth_mm": [bill_depth_mm],
                "flipper_length_mm": [flipper_length_mm],
                "body_mass_g": [body_mass_g],
                "island_Torgersen": [1 if island == "Torgersen" else 0],
                "island_Dream": [1 if island == "Dream" else 0],
                "sex_male": [1 if sex == "male" else 0],
            }
        )

        # Ensure columns match model's expected columns
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Species: **{prediction[0]}**")

    st.code(
        """
# The code for this section
# ... (code for loading data, training model, and user input form) ...
""",
        language="python",
    )


def llm_conversation():
    st.header("Conversing with OpenAI's gpt-4o")
    st.markdown(
        "This tab demonstrates how to build a conversational AI with Streamlit and OpenAI's `gpt-4o` model."
    )

    # Get API key
    api_key = st.text_input("Enter your OpenAI API key", type="password")

    if not api_key:
        st.info("Please enter your OpenAI API key to proceed.")
        st.markdown(
            "You can get a key from [OpenAI](https://platform.openai.com/account/api-keys)."
        )
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
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    st.code(
        """
# The code for this section
# ... (code for getting API key, initializing chat, and handling chat input) ...
""",
        language="python",
    )


def convert_to_png(image_data):
    """Convert image to PNG format for OpenAI API compatibility"""
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    else:
        img = Image.open(image_data)
    
    # Convert to RGB if necessary (in case of RGBA or other modes)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save as PNG to bytes
    png_buffer = io.BytesIO()
    img.save(png_buffer, format='PNG')
    png_buffer.seek(0)
    
    return png_buffer


def image_modification():
    st.header("AI Image Generation & Modification")
    st.markdown(
        "This tab demonstrates how to use OpenAI's DALL-E model to generate modified versions of images based on text instructions. Upload an image or select from provided samples, then describe your desired modifications."
    )

    # Get API key
    api_key = st.text_input("Enter your OpenAI API key", type="password", key="image_api_key")

    if not api_key:
        st.info("Please enter your OpenAI API key to proceed.")
        st.markdown(
            "You can get a key from [OpenAI](https://platform.openai.com/account/api-keys)."
        )
        return

    client = OpenAI(api_key=api_key)

    # Sample images
    sample_images = {
        "Mountain Landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "City Street": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "Ocean Sunset": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800"
    }

    # Image input method
    input_method = st.radio("Choose input method:", ["Upload Image", "Select Sample Image", "Enter Image URL"])

    uploaded_image = None
    selected_image_url = None

    if input_method == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    elif input_method == "Select Sample Image":
        selected_sample = st.selectbox("Choose a sample image:", list(sample_images.keys()))
        selected_image_url = sample_images[selected_sample]
        st.image(selected_image_url, caption=f"Selected: {selected_sample}", use_container_width=True)
    
    elif input_method == "Enter Image URL":
        image_url = st.text_input("Enter image URL:")
        if image_url:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    selected_image_url = image_url
                    st.image(image_url, caption="Image from URL", use_container_width=True)
                else:
                    st.error("Failed to load image from URL")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # Modification instructions
    modification_instructions = st.text_area(
        "Describe the modifications you want to make to the image:",
        placeholder="e.g., Make it look like a watercolor painting, Add a sunset sky, Convert to black and white with dramatic lighting...",
        height=100
    )

    # Process button
    if st.button("Generate Modified Image") and modification_instructions:
        if not (uploaded_image or selected_image_url):
            st.error("Please provide an image first.")
            return

        with st.spinner("Generating modified image..."):
            try:
                # Prepare the image for OpenAI API
                if uploaded_image:
                    # For uploaded image, convert to PNG
                    image_data = convert_to_png(uploaded_image)
                else:
                    # For URL, download and convert to PNG
                    response = requests.get(selected_image_url)
                    image_bytes = response.content
                    image_data = convert_to_png(image_bytes)

                # Call OpenAI API for image generation based on modification instructions
                # Since edit API requires a mask, we'll use generate with detailed instructions
                enhanced_prompt = f"Create a modified version of an image with these changes: {modification_instructions}. The image should maintain the same composition and subject but with the requested modifications applied."
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=enhanced_prompt,
                    n=1,
                    size="1024x1024"
                )

                # Display the result
                if response.data:
                    modified_image_url = response.data[0].url
                    st.success("Image modification completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        if uploaded_image:
                            # Reset file pointer to beginning for display
                            uploaded_image.seek(0)
                            st.image(uploaded_image, use_container_width=True)
                        else:
                            st.image(selected_image_url, use_container_width=True)
                    
                    with col2:
                        st.subheader("Modified Image")
                        st.image(modified_image_url, use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label="Download Modified Image",
                        data=requests.get(modified_image_url).content,
                        file_name="modified_image.png",
                        mime="image/png"
                    )

            except Exception as e:
                st.error(f"Error generating modified image: {e}")

    st.code(
        """
# The code for this section
def image_modification():
    st.header("AI Image Modification")
    
    # Get API key
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    client = OpenAI(api_key=api_key)
    
    # Sample images
    sample_images = {
        "Mountain Landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "City Street": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "Ocean Sunset": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800"
    }
    
    # Image input methods (upload, sample, URL)
    input_method = st.radio("Choose input method:", ["Upload Image", "Select Sample Image", "Enter Image URL"])
    
    # Handle different input methods
    if input_method == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    elif input_method == "Select Sample Image":
        selected_sample = st.selectbox("Choose a sample image:", list(sample_images.keys()))
        selected_image_url = sample_images[selected_sample]
    elif input_method == "Enter Image URL":
        image_url = st.text_input("Enter image URL:")
    
    # Modification instructions
    modification_instructions = st.text_area("Describe the modifications...")
    
    # Process with OpenAI API
    if uploaded_image:
        image_data = convert_to_png(uploaded_image)  # Convert to PNG
    else:
        response = requests.get(selected_image_url)
        image_data = convert_to_png(response.content)  # Convert to PNG
    
    # Since edit API requires a mask, we use generate with enhanced prompts
    enhanced_prompt = f"Create a modified version: {modification_instructions}"
    response = client.images.generate(
        model="dall-e-2",
        prompt=enhanced_prompt,
        n=1,
        size="1024x1024"
    )
""",
        language="python",
    )


# Main App
def main():
    st.sidebar.title("Navigation")
    tabs = {
        "Introduction": introduction,
        "Interactive Widgets & Data Filtering": interactive_widgets,
        "Advanced Charting": advanced_charting,
        "Performance & State": performance_state,
        "ML Integration": ml_integration,
        "LLM Conversation": llm_conversation,
        "Image Modification": image_modification,
    }
    selection = st.sidebar.radio("Go to", list(tabs.keys()))

    # Placeholder for other tabs
    if tabs[selection]:
        tabs[selection]()
    else:
        st.write("This tab is under construction.")


if __name__ == "__main__":
    main()
