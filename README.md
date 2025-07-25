# Streamlit Capabilities Demo

This repository contains a comprehensive Streamlit application designed to demonstrate its core capabilities, particularly for data science and machine learning workflows. The application is self-documenting, showcasing the Python code used to generate each component.

The target audience is experienced Python programmers and machine learning engineers who are new to Streamlit and want a quick, hands-on tour of its most powerful features.

## Features Demonstrated

The application is organized into several tabs, each focusing on a different aspect of Streamlit:

*   **Introduction:** A basic overview of Streamlit and how to display text and code.
*   **Interactive Widgets & Data Filtering:** Using widgets like sliders, dropdowns, and multi-select boxes to interactively filter a Pandas DataFrame.
*   **Advanced Charting:** Integrating with Plotly to create dynamic, publication-quality charts that respond to user input.
*   **Performance & State:**
    *   **Caching:** Using `st.cache_data` and `st.cache_resource` to optimize performance by caching expensive function calls and pre-trained models.
    *   **Session State:** Using `st.session_state` to maintain user session information across reruns, enabling more complex, stateful applications.
*   **Machine Learning Integration:**
    *   Building a complete ML prediction app using a pre-trained scikit-learn model.
    *   Using `st.form` to batch user inputs for a model prediction.

*   **LLM Conversation:** A chat interface to converse with OpenAI's `gpt-4o` model.

*   **Image Modification:** An AI-powered image editing interface using OpenAI's DALL-E model:
    *   Upload images, select from sample images, or provide image URLs
    *   Provide text instructions for image modifications
    *   View side-by-side comparison of original and modified images
    *   Download the modified images

The demo uses the **Palmer Penguins** dataset, a popular and clean dataset for classification tasks, to showcase data filtering, visualization, and model prediction.

## Installation

1.  Clone this repository or download the source code.
2.  Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application, execute the following command in your terminal from the project's root directory:

```bash
streamlit run app.py
```

This will start the local Streamlit server and open the application in your default web browser.
