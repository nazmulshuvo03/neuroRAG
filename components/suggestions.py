"""Suggestions component with clickable question buttons"""
import streamlit as st


# Sample suggestions focused on Neurodevelopmental Disorders
SUGGESTIONS = [
    "What is the definition of a Neurodevelopmental Disorder?",
    "What are the core symptoms of Autism Spectrum Disorder (ASD)?",
    "Briefly explain what ADHD is.",
    "What is Dyslexia?",
    "What is the estimated prevalence of autism according to the documents?",
    "Are there specific genetic factors linked to neurodevelopmental disorders?",
    "What are the common comorbidities associated with ADHD?",
    "What does the text say about early intervention strategies?",
    "How do the symptoms of ADHD differ from those of Autism?",
    "Compare the treatment approaches for Dyslexia vs. Dyscalculia.",
    "What is the relationship between environmental factors and neurodevelopmental disorders?",
    "What are the diagnostic criteria for Autism Spectrum Disorder?",
    "How does early intervention impact children with neurodevelopmental disorders?",
    "What role does genetics play in ADHD?",
    "What are the characteristics of Dyscalculia?",
]


def render_suggestions():
    """Render the suggestions section with clickable buttons"""
    st.markdown("### 💡 Suggestions")
    st.markdown("Click on any question below to populate the chat input:")
    
    # Initialize session state for selected suggestion
    if "selected_suggestion" not in st.session_state:
        st.session_state.selected_suggestion = None
    
    # Style the buttons to be full width and properly formatted
    st.markdown(
        """
        <style>
        /* Make suggestion buttons full width and left-aligned */
        div.stButton > button {
            width: 100%;
            text-align: left;
            white-space: normal;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Create a scrollable container using streamlit's built-in container with fixed height
    # Use st.container with height parameter (available in newer Streamlit versions)
    container = st.container(height=600)
    
    with container:
        for i, question in enumerate(SUGGESTIONS):
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                st.session_state.selected_suggestion = question
                # Scroll to chatbox by triggering rerun
                st.rerun()
