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
    st.header("Advanced AI Image Modification & Generation")
    st.markdown(
        "This tab demonstrates advanced image modification using OpenAI's latest models. Choose from multiple approaches including DALL-E 3, image variations, and custom modifications."
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

    # Sample images with better variety
    sample_images = {
        "Mountain Landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "City Street": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "Ocean Sunset": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800",
        "Portrait": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
        "Abstract Art": "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=800"
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

    # Image Modification Options
    st.subheader("Modification Options")
    
    modification_type = st.selectbox(
        "Choose modification type:",
        [
            "New Image Generation",
            "Style Transfer",
            "Background Enhancement",
            "Creative Reimagining"
        ]
    )

    # Custom instructions based on modification type
    if modification_type == "New Image Generation":
        modification_instructions = st.text_area(
            "Describe the new image you want to generate:",
            placeholder="e.g., A futuristic cityscape with flying cars and neon lights, A mountain landscape in watercolor style within a Studio Ghibli scene...",
            height=100
        )
        st.info("💡 Best for creating completely new images. Works great with detailed descriptions.")
        
    elif modification_type == "Style Transfer":
        style_options = ["Watercolor", "Oil Painting", "Pencil Sketch", "Pop Art", "Vintage Film", "Minimalist", "Cyberpunk", "Studio Ghibli", "Impressionist", "Abstract"]
        selected_style = st.selectbox("Choose artistic style:", style_options)
        modification_instructions = f"Convert the image to {selected_style} style"
        st.info("💡 Good for artistic style changes. Works well with most image types.")
        
    elif modification_type == "Background Enhancement":
        background_options = ["Golden Hour", "Studio", "Nature", "City", "Fantasy", "Minimalist", "Sunset", "Mountain", "Beach", "Forest"]
        selected_background = st.selectbox("Choose new background:", background_options)
        modification_instructions = f"Replace the background with a {selected_background} scene"
        st.info("💡 Good for background replacement. Works best with clear subjects.")
        
    elif modification_type == "Creative Reimagining":
        creative_options = ["Studio Ghibli Style", "Cyberpunk", "Steampunk", "Fairy Tale", "Sci-Fi", "Retro 80s", "Vintage", "Futuristic"]
        selected_creative = st.selectbox("Choose creative direction:", creative_options)
        modification_instructions = f"Transform the image into {selected_creative} style"
        st.info("💡 Perfect for creating unique, artistic interpretations!")

    # Generation Settings
    with st.expander("⚙️ Generation Settings"):
        st.markdown("**DALL-E 3 Settings:**")
        
        # Always use DALL-E 3
        model_choice = "dall-e-3"
        st.info("💡 Using DALL-E 3: Best quality, generates 1 image, supports different aspect ratios")
        
        size_choice = st.selectbox(
            "Size:",
            ["1024x1024", "1792x1024", "1024x1792"],
            help="Square (1024x1024) for most uses, Wide (1792x1024) for landscapes, Tall (1024x1792) for portraits"
        )
        
        quality_choice = st.selectbox(
            "Quality:",
            ["standard", "hd"],
            help="Standard: Good quality, cheaper. HD: Higher quality, more expensive"
        )
        
        st.markdown("**Creativity Control:**")
        creativity_level = st.selectbox(
            "Stay close to original:",
            ["More Creative", "Balanced", "Stay Close"],
            help="More Creative: More artistic freedom. Stay Close: More faithful to original"
        )
        
        num_variations = 1

    # Process button
    if st.button("Generate Modified Image", type="primary") and modification_instructions:
        if not (uploaded_image or selected_image_url):
            st.error("Please provide an image first.")
            return

        with st.spinner("Generating modified image..."):
            try:
                # Prepare the image for OpenAI API
                if uploaded_image:
                    image_data = convert_to_png(uploaded_image)
                else:
                    response = requests.get(selected_image_url)
                    image_bytes = response.content
                    image_data = convert_to_png(image_bytes)

                # Enhanced prompt based on modification type and creativity level
                if modification_type == "New Image Generation":
                    if creativity_level == "Stay Close":
                        enhanced_prompt = f"Create a new image based on: {modification_instructions}. Keep the same composition, colors, and style as the original image."
                    elif creativity_level == "Balanced":
                        enhanced_prompt = f"Create a new image based on: {modification_instructions}. Maintain some elements from the original while being creative."
                    else:  # More Creative
                        enhanced_prompt = modification_instructions
                        
                elif modification_type == "Style Transfer":
                    if creativity_level == "Stay Close":
                        enhanced_prompt = f"Apply {modification_instructions} while keeping the original composition, subject, and details very close to the source image."
                    elif creativity_level == "Balanced":
                        enhanced_prompt = f"Apply {modification_instructions} while maintaining the same composition and subject."
                    else:  # More Creative
                        enhanced_prompt = f"Apply {modification_instructions} with artistic freedom and creative interpretation."
                        
                elif modification_type == "Background Enhancement":
                    if creativity_level == "Stay Close":
                        enhanced_prompt = f"Replace the background with a {modification_instructions} while keeping the main subject exactly as it is in the original image."
                    elif creativity_level == "Balanced":
                        enhanced_prompt = f"Replace the background with a {modification_instructions} while maintaining the same subject and composition."
                    else:  # More Creative
                        enhanced_prompt = f"Replace the background with a {modification_instructions} and feel free to enhance the overall scene creatively."
                        
                elif modification_type == "Creative Reimagining":
                    if creativity_level == "Stay Close":
                        enhanced_prompt = f"Transform the image into {modification_instructions} while keeping the original composition and subject very close to the source image."
                    elif creativity_level == "Balanced":
                        enhanced_prompt = f"Transform the image into {modification_instructions} while maintaining the same composition and subject."
                    else:  # More Creative
                        enhanced_prompt = f"Transform the image into {modification_instructions} with artistic freedom and creative interpretation."

                # Call OpenAI API with DALL-E 3
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=enhanced_prompt,
                    n=1,
                    size=size_choice,
                    quality=quality_choice
                )

                # Display the result
                if response.data:
                    st.success(f"Image modification completed! Generated {len(response.data)} image(s).")
                    
                    # Display original image
                    st.subheader("Original Image")
                    if uploaded_image:
                        uploaded_image.seek(0)
                        st.image(uploaded_image, use_container_width=True)
                    else:
                        st.image(selected_image_url, use_container_width=True)
                    
                    # Display generated images
                    st.subheader("Generated Image(s)")
                    
                    if len(response.data) == 1:
                        # Single image - display side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(response.data[0].url, use_container_width=True, caption="Generated Image")
                        with col2:
                            st.download_button(
                                label="Download Image",
                                data=requests.get(response.data[0].url).content,
                                file_name="generated_image.png",
                                mime="image/png"
                            )
                    else:
                        # Multiple images - display in grid
                        cols = st.columns(min(len(response.data), 3))
                        for i, image_data in enumerate(response.data):
                            with cols[i % 3]:
                                st.image(image_data.url, use_container_width=True, caption=f"Variation {i+1}")
                                st.download_button(
                                    label=f"Download Variation {i+1}",
                                    data=requests.get(image_data.url).content,
                                    file_name=f"generated_image_{i+1}.png",
                                    mime="image/png"
                                )

            except Exception as e:
                st.error(f"Error generating modified image: {e}")
                st.info("💡 Tip: Make sure your OpenAI API key has sufficient credits and the model you selected is available.")

    st.code(
        """
# The code for this section
def image_modification():
    st.header("Advanced AI Image Modification")
    
    # Get API key
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    client = OpenAI(api_key=api_key)
    
    # Sample images with better variety
    sample_images = {
        "Mountain Landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "City Street": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "Ocean Sunset": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800",
        "Portrait": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
        "Abstract Art": "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=800"
    }
    
    # Advanced modification options
    modification_type = st.selectbox("Choose modification type:", [
        "DALL-E 3 - Advanced Generation",
        "DALL-E 3 - Image Variations", 
        "DALL-E 2 - Custom Modifications",
        "Style Transfer",
        "Background Replacement",
        "Object Addition/Removal"
    ])
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        model_choice = st.selectbox("Select model:", ["dall-e-3", "dall-e-2"])
        size_choice = st.selectbox("Select image size:", ["1024x1024", "1792x1024", "1024x1792"])
        quality_choice = st.selectbox("Select quality:", ["standard", "hd"])
        num_variations = st.slider("Number of variations:", 1, 4, 1)
    
    # Enhanced prompt generation based on modification type
    if modification_type == "DALL-E 3 - Advanced Generation":
        enhanced_prompt = modification_instructions
    elif modification_type == "DALL-E 3 - Image Variations":
        enhanced_prompt = f"Create variations of this image with these changes: {modification_instructions}"
    else:
        enhanced_prompt = f"Apply {modification_instructions} to this image"
    
    # Call OpenAI API with selected model and settings
    if model_choice == "dall-e-3":
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            n=1,
            size=size_choice,
            quality=quality_choice
        )
    else:
        response = client.images.generate(
            model="dall-e-2",
            prompt=enhanced_prompt,
            n=num_variations,
            size=size_choice
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
