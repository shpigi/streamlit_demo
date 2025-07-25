import streamlit as st

# Import section modules
from sections.introduction import introduction
from sections.interactive_widgets import interactive_widgets
from sections.advanced_charting import advanced_charting
from sections.performance_state import performance_state
from sections.ml_integration import ml_integration
from sections.llm_conversation import llm_conversation
from sections.image_modification import image_modification

def display_code_file(filename, title):
    """Display a code file with syntax highlighting"""
    try:
        with open(f"sections/{filename}", 'r') as f:
            code_content = f.read()
        
        st.subheader(f"📄 {title}")
        st.code(code_content, language="python")
        
    except FileNotFoundError:
        st.error(f"Code file {filename} not found")
    except Exception as e:
        st.error(f"Error reading code file: {e}")

# Page Configuration
st.set_page_config(
    page_title="Streamlit Demo",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded",
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

    # Run the selected function
    if tabs[selection]:
        tabs[selection]()
        
        # Display the corresponding code file
        code_files = {
            "Introduction": ("introduction.py", "Introduction Code"),
            "Interactive Widgets & Data Filtering": ("interactive_widgets.py", "Interactive Widgets Code"),
            "Advanced Charting": ("advanced_charting.py", "Advanced Charting Code"),
            "Performance & State": ("performance_state.py", "Performance & State Code"),
            "ML Integration": ("ml_integration.py", "ML Integration Code"),
            "LLM Conversation": ("llm_conversation.py", "LLM Conversation Code"),
            "Image Modification": ("image_modification.py", "Image Modification Code"),
        }
        
        if selection in code_files:
            filename, title = code_files[selection]
            display_code_file(filename, title)
    else:
        st.write("This tab is under construction.")

if __name__ == "__main__":
    main()
