import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Data Doctor",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🩺 Data Doctor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a messy CSV. Get clean data in seconds.</div>', unsafe_allow_html=True)

API_URL = "https://datadoctor-api.onrender.com"

uploaded_file = st.file_uploader(
    "Drop your CSV here",
    type="csv",
    help="Upload any CSV with missing values, duplicates, or messy formatting."
)

if uploaded_file:
    st.subheader("📋 Original Data")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df)} rows × {len(df.columns)} columns")

    if st.button("🧹 Clean My Data", type="primary", use_container_width=True):
        with st.spinner("Cleaning your data... (may take 30s on first run if API is cold)"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                response = requests.post(f"{API_URL}/clean/", files=files, timeout=90)
                response.raise_for_status()
                result = response.json()

                st.success("✅ Data cleaned successfully!")

                d = result["diagnosis"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", d["num_rows"])
                m2.metric("Columns", d["num_columns"])
                m3.metric("Duplicates Removed", d["duplicate_rows"])
                m4.metric("Missing Values Fixed", sum(d["missing_values"].values()))

                st.subheader("✨ Cleaned Data")
                cleaned = pd.DataFrame(result["cleaned_preview"])
                st.dataframe(cleaned, use_container_width=True)
                st.caption(f"{result['cleaned_row_count']} rows × {result['cleaned_column_count']} columns")

                st.download_button(
                    "⬇️ Download Cleaned CSV",
                    data=cleaned.to_csv(index=False),
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                with st.expander("🔍 Full Diagnosis Report"):
                    st.json(d)

            except requests.exceptions.Timeout:
                st.error("The API took too long. It may be waking up. Try again in 30 seconds.")
            except Exception as e:
                st.error(f"Error: {e}")