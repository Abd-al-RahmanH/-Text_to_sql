import os
import streamlit as st
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Hardcoded credentials (use for testing purposes only)
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "Q2uZJDscw55ZJ6IGrpOyHw4c7RpkJyY-z6GKIH5Qj--s"
}

project_id = "8e75587d-5ef2-4d94-a5d1-9493d6145ac3"

# Set up Watsonx.ai model configuration
parameters = {
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 196,
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.TEMPERATURE: 0.3,
    GenParams.REPETITION_PENALTY: 1
}

# Initialize Watsonx.ai model
model = Model(
    model_id="granite-34b-code-instruct",
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Streamlit UI setup
st.set_page_config(page_title="NLP to SQL Converter", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.title("NLP-to-SQL Converter")
    st.markdown("### Watsonx.ai Settings")
    decoding_method = st.selectbox("Decoding Method", ["Greedy", "Sample"], index=0)
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=400, value=196, step=10)

    # Update parameters dynamically
    parameters[GenParams.DECODING_METHOD] = decoding_method.lower()
    parameters[GenParams.MAX_NEW_TOKENS] = max_tokens

# Main Page
st.title("Natural Language to SQL Query Generator")
st.markdown("Convert natural language queries into SQL statements using **IBM Watsonx.ai**.")

# Input box for natural language query
user_query = st.text_area("Enter your query in natural language:", placeholder="e.g., Find all employees who joined in 2023.")

# Button to generate SQL
if st.button("Generate SQL"):
    if not user_query.strip():
        st.error("Please enter a query to generate SQL.")
    else:
        with st.spinner("Generating SQL..."):
            try:
                # Generate SQL using Watsonx.ai
                response = model.generate_text(prompt=user_query)
                # Extract generated SQL from the response
                if isinstance(response, str):  # Handle response if it's a string
                    response_text = response
                elif "results" in response:  # Handle structured response
                    response_text = response["results"][0]["generated_text"]
                else:
                    response_text = "Error: Unexpected response format."
                
                st.markdown("### 🧾 Generated SQL")
                st.code(response_text, language="sql")
            except Exception as e:
                st.error(f"Error generating SQL: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by **[Your Name]** | Powered by **IBM Watsonx.ai**")
