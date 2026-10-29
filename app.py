import streamlit as st
from pipeline import run_research_pipeline

st.set_page_config(
    page_title="Multi-Agent AI Research",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Multi-Agent AI Research System")
st.caption(
    "Powered by local Ollama, Tavily Search, "
    "Reader Agent, Writer Agent and Critic Agent"
)

st.info(
    "Research results AI-generated hain. "
    "Important facts aur source links ko independently verify karein."
)

with st.sidebar:
    st.header("⚙️ Research Settings")
    st.write("Model: Local Ollama")
    st.write("Search: Tavily")
    st.write("Agents: Search · Reader · Writer · Critic")
    st.caption("Tavily API key .env file mein rakhein.")

st.subheader("What do you want to research?")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. History of cricket"
    )

    submitted = st.form_submit_button(
        "🚀 Start Research",
        use_container_width=True
    )

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            with st.spinner(
                "Agents are researching... Please wait."
            ):
                result = run_research_pipeline(topic.strip())

            st.success("Research pipeline completed!")

            st.subheader("📄 Research Report")
            report = result.get("report", "")

            if report:
                st.markdown(report)
                st.download_button(
                    "⬇️ Download Report",
                    data=str(report),
                    file_name="research_report.md",
                    mime="text/markdown"
                )
            else:
                st.warning("Writer Agent did not return a report.")

            with st.expander("🔎 Search Results"):
                st.text(
                    str(result.get("search_results", "No results"))
                )

            with st.expander("📚 Scraped Content"):
                st.text(
                    str(result.get("scraped_content", "No content"))
                )

            with st.expander("🧐 Critic Feedback"):
                st.markdown(
                    str(result.get("feedback", "No feedback"))
                )

            with st.expander("🔗 Verification Reminder"):
                st.write(
                    "Check that every citation URL exists and "
                    "actually supports the related claim. "
                    "A generated citation is not proof of a source."
                )

        except Exception as error:
            st.error("Research pipeline failed.")
            st.code(str(error))
            st.write(
                "Check that Ollama is running, the required "
                "packages are installed, and your .env file "
                "contains a valid Tavily API key."
            )