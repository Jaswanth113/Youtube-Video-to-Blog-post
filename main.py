import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
code_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)

class VideoState(TypedDict):
    video_id: str
    transcript: str
    summary: Optional[str]
    blog_post: Optional[str]

def get_transcript_node(state: VideoState) -> VideoState:
    video_id = state["video_id"]
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        transcript = " ".join([t["text"] for t in fetched_transcript.to_raw_data()])
        return {"transcript": transcript}
    except Exception as e:
        st.error(f"could not fetch the transcript for video ID {video_id}: {e}")
        return {"transcript": ""}

def summarize_transcript_node(state: VideoState) -> VideoState:
    transcript = state.get("transcript", "")
    if not transcript:
        return {"summary": "No transcript was available to summarize."}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False,
    )
    
    texts = text_splitter.create_documents([transcript])
    
    summaries = []
    for i, doc in enumerate(texts):
        prompt = f"Summarize the following text concisely, focusing on the main topics and key takeaways:\n\n{doc.page_content}"
        response = llm.invoke(prompt)
        summaries.append(response.content)
        
    combined_summary = " ".join(summaries)
    
    if len(summaries) > 1:
        final_prompt = f"Combine and refine the following summaries into a single, coherent, and concise summary:\n\n{combined_summary}"
        final_response = llm.invoke(final_prompt)
        return {"summary": final_response.content}
    else:
        return {"summary": combined_summary}

def generate_blog_node(state: VideoState) -> VideoState:
    st.write("Generating final blog post...")
    summary = state.get("summary", "")
    
    prompt = f"""
    You are a skilled blog writer. Using the provided YouTube video summary, generate a well-structured blog post with the following sections only:

    1. **Title** - a catchy, reader-friendly headline (7-12 words).  
    2. **Description** - a clear and engaging explanation that expands on the summary, highlighting the main ideas in a natural flow (2–3 paragraphs).  
    3. **Conclusion** - a concise closing that reinforces the key points and leaves the reader with a final takeaway.  

    Video Summary:
    {summary}
    """
    response = code_llm.invoke(prompt)
    return {"blog_post": response.content}

#langgraph structure
workflow = StateGraph(VideoState)

workflow.add_node("fetch_transcript", get_transcript_node)
workflow.add_node("summarize_transcript", summarize_transcript_node)
workflow.add_node("generate_blog", generate_blog_node)
workflow.set_entry_point("fetch_transcript")
workflow.add_edge("fetch_transcript", "summarize_transcript")
workflow.add_edge("summarize_transcript", "generate_blog")
workflow.add_edge("generate_blog", END)

state_machine = workflow.compile()

#streamlit
st.title("YouTube video to Blog post")

url = st.text_input("Enter YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=your_video_id")

if st.button("Process Video"):
    if url and "v=" in url:
        video_id = url.split("v=")[-1].split("&")[0]
        
        with st.spinner("Processing video: Fetching transcript, summarizing, and generating blog post..."):
            final_result = state_machine.invoke({"video_id": video_id})
            st.session_state.blog_post = final_result.get("blog_post")
            st.session_state.reviewed_summary = final_result.get("summary")
        st.success("Blog post generated!")
    else:
        st.error("Please enter a valid YouTube URL.")

if "blog_post" in st.session_state:
    st.header("Your Generated Blog Post")
    st.markdown(st.session_state.blog_post)