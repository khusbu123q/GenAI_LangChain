from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="Research Tool", layout="centered")

st.title("🔍 Research Tool")
st.write("Enter any text and get a quick AI summary")

user_input = st.text_area("Enter your prompt", height=150)

# ---------------- BUTTON ----------------
if st.button("Summarize"):
    if user_input.strip():

        try:
            with st.spinner("Thinking..."):

                # Create LLM ONLY when needed (prevents crash)
                llm = HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-72B-Instruct",
                    task="text-generation",
                )

                model = ChatHuggingFace(llm=llm)

                # Better prompt (more structured)
                prompt = f"Summarize the following in simple terms:\n\n{user_input}"

                result = model.invoke(prompt)

                st.subheader("📄 Summary")
                st.write(result.content)

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter a prompt")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit + Hugging Face")