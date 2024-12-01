import os
import pandas as pd
import sqlite3
import streamlit as st
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Hardcoded Watsonx.ai credentials and configurations
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "Q2uZJDscw55ZJ6IGrpOyHw4c7RpkJyY-z6GKIH5Qj--s"
}
project_id = "8e75587d-5ef2-4d94-a5d1-9493d6145ac3"
model_id = "meta-llama/llama-3-405b-instruct"  # Replace with your Watsonx.ai model ID
parameters = {
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 196,
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.TEMPERATURE: 0.3,
    GenParams.REPETITION_PENALTY: 1
}

# Initialize Watsonx.ai model
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Streamlit UI
st.title("NLP to SQL Converter")
st.sidebar.header("Model Parameters")

# Sidebar parameters
max_tokens = st.sidebar.slider("Max Tokens", 10, 4000, value=150)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.3)

# Update parameters dynamically
parameters[GenParams.MAX_NEW_TOKENS] = max_tokens
parameters[GenParams.TEMPERATURE] = temperature

# User input for natural language query
user_query = st.text_input("Enter your query (e.g., 'Which employee had the most sales in October 2023'):")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Convert DataFrame to SQLite database
    conn = sqlite3.connect(":memory:")
    df.to_sql("uploaded_data", conn, index=False, if_exists="replace")

    if st.button("Generate and Execute SQL"):
        if not user_query.strip():
            st.error("Please enter a query to generate SQL.")
        else:
            with st.spinner("Generating SQL and querying the dataset..."):
                try:
                    # Generate SQL query using Watsonx.ai
                    prompt = f"""
                    You are an SQL assistant.
                    - The table name is `uploaded_data`.
                    - Column names: {', '.join([f'"{col}"' for col in df.columns])}.
                    - Generate an SQL query for the following user query: "{user_query}".
                    Ensure proper escaping for column names with spaces or special characters.
                    """
                    response = model.generate_text(prompt=prompt)

                    # Extract the generated SQL query
                    if isinstance(response, str):
                        sql_query = response.strip()
                    elif "results" in response:
                        sql_query = response["results"][0]["generated_text"].strip()
                    else:
                        sql_query = "Error: Unexpected response format."

                    # Display generated SQL query
                    st.markdown("### 🧾 Generated SQL")
                    st.code(sql_query, language="sql")

                    # Execute the SQL query
                    try:
                        query_result = pd.read_sql_query(sql_query, conn)
                        st.markdown("### 📊 Query Results")
                        st.dataframe(query_result)
                    except Exception as sql_error:
                        st.error(f"Error executing SQL query: {sql_error}")
                except Exception as e:
                    st.error(f"Error generating SQL: {e}")
