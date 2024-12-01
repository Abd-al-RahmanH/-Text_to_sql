import streamlit as st
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods, ModelTypes

# App title
st.set_page_config(page_title="NLP to SQL Converter", layout="wide")
st.title("🔄 NLP to SQL Converter")
st.write("Easily convert natural language queries into SQL using Watsonx.ai models.")

# Sidebar: Model selection and parameters
with st.sidebar:
    st.header("🛠 Configuration")
    
    # Model Selection
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Choose a Watsonx Model",
        [
            "GRANITE_34B_CODE_INSTRUCT",
            "META-LLAMA/LLAMA-3-405B-INSTRUCT",
            "CODELLAMA/CODELLAMA-34B-INSTRUCT-HF",
        ],
    )
    
    # Generation Parameters
    st.subheader("Generation Parameters")
    max_new_tokens = st.slider("Max Tokens", min_value=10, max_value=4000, value=150)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    decoding_method = st.radio(
        "Decoding Method", options=[DecodingMethods.GREEDY.value, DecodingMethods.SAMPLE.value]
    )

# Watsonx Model Configuration
params = {
    GenParams.DECODING_METHOD: decoding_method,
    GenParams.MAX_NEW_TOKENS: max_new_tokens,
    GenParams.TEMPERATURE: temperature,
    GenParams.REPETITION_PENALTY: 1,
}
credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": "Q2uZJDscw55ZJ6IGrpOyHw4c7RpkJyY-z6GKIH5Qj--s"}
project_id = "8e75587d-5ef2-4d94-a5d1-9493d6145ac3"

# Load Watsonx Model
st.info("Loading Watsonx.ai model...")
model = Model(model_id=ModelTypes[selected_model], params=params, credentials=credentials, project_id=project_id)

if model:
    st.success(f"Model '{selected_model}' loaded successfully!")

# Center page for query input
st.markdown("### 📝 Input Your Query")
user_query = st.text_area(
    "Enter your natural language query below (e.g., 'Which employee had the most sales in October 2023?')",
    height=150,
)

# SQL Generation Button
if st.button("Generate SQL"):
    if not user_query.strip():
        st.error("Please enter a query to generate SQL.")
    else:
        with st.spinner("Generating SQL..."):
            try:
                response = model.generate_text(prompt=user_query)
                st.markdown("### 🧾 Generated SQL")
                st.code(response["results"][0]["generated_text"], language="sql")
            except Exception as e:
                st.error(f"Error generating SQL: {e}")
