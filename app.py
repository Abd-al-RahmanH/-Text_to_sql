import pandas as pd
import sqlite3
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
st.set_page_config(page_title="Talk-to-CSV with NLP-to-SQL", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Talk-to-CSV Settings")
    st.markdown("### Watsonx.ai Configuration")
    decoding_method = st.selectbox("Decoding Method", ["Greedy", "Sample"], index=0)
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=400, value=196, step=10)

    # Update parameters dynamically
    parameters[GenParams.DECODING_METHOD] = decoding_method.lower()
    parameters[GenParams.MAX_NEW_TOKENS] = max_tokens

st.title("Talk-to-CSV: NLP-to-SQL Querying on Your Data")
st.markdown("Upload a CSV file and use natural language queries to interact with your data.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Read and display the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        st.success(f"CSV file '{uploaded_file.name}' uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        df = None

    # If the file is loaded successfully
    if df is not None:
        # Save the dataframe to a temporary SQLite database
        conn = sqlite3.connect(":memory:")
        df.to_sql("uploaded_data", conn, index=False, if_exists="replace")

        # Input box for natural language query
        user_query = st.text_area("Enter your query in natural language:", placeholder="e.g., Find employees with salaries over $50,000.")

        if st.button("Generate and Execute SQL"):
            if not user_query.strip():
                st.error("Please enter a query to generate SQL.")
            else:
                with st.spinner("Generating SQL and querying the dataset..."):
                    try:
                        # Generate SQL using Watsonx.ai
                        prompt = f"Generate an SQL query based on this table schema: {list(df.columns)}. User's query: {user_query}"
                        response = model.generate_text(prompt=prompt)

                        # Extract generated SQL from the response
                        if isinstance(response, str):  # Handle response if it's a string
                            sql_query = response.strip()
                        elif "results" in response:  # Handle structured response
                            sql_query = response["results"][0]["generated_text"].strip()
                        else:
                            sql_query = "Error: Unexpected response format."

                        # Display generated SQL
                        st.markdown("### 🧾 Generated SQL")
                        st.code(sql_query, language="sql")

                        # Execute the SQL query on the SQLite database
                        try:
                            query_result = pd.read_sql_query(sql_query, conn)
                            st.markdown("### 📊 Query Results")
                            st.dataframe(query_result)
                        except Exception as sql_error:
                            st.error(f"Error executing SQL query: {sql_error}")
                    except Exception as e:
                        st.error(f"Error generating SQL: {e}")

        # Close the SQLite connection
        conn.close()

# Footer
st.markdown("---")
st.markdown("Developed by **[Your Name]** | Powered by **IBM Watsonx.ai**")
