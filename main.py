import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.7)
code_llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it", temperature=0.2)

class VideoState(TypedDict):
    video_id: str
    transcript: str
    summary: Optional[str]
    blog_post: Optional[str]

def extract_video_id(url: str) -> Optional[str]:

    regex_patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

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
        chunk_size=2000,
        chunk_overlap=200,
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
    transcript = state.get("transcript", "")
    
    prompt = f"""
    You are a friendly and insightful productivity blogger. Your goal is to write a blog post that feels personal, actionable, and encouraging.

    **YOUR OUTPUT MUST FOLLOW THIS MARKDOWN FORMAT EXACTLY, INCLUDING THE BLANK LINES BETWEEN SECTIONS:**

    **Title:**  
    [Write a clear and professional headline that directly reflects the main topic of the YouTube video. Keep it straightforward, natural, and within 7-12 words.]

    **Description:**  
    [Write 3-5 detailed paragraphs expanding on the video content. Summarize the key points from the transcript in a smooth, natural flow. Add context, examples, or insights where relevant to make it engaging and easy to read. Avoid robotic repetition — make it feel like a professional blog article. Ensure paragraphs connect logically and the writing feels polished.]

    **Conclusion:**  
    [Write a single, detailed closing section of 1-3 paragraphs. Provide a thoughtful summary of the video's main message. Reinforce why it matters or how the reader can apply it. End with a motivating or insightful statement that leaves the reader with a strong takeaway.]

    ---
    **SOURCE MATERIAL:**

    **Video Summary:**
    {summary}

    **Full Transcript:**
    {transcript}
    ---
    """
    response = code_llm.invoke(prompt)
    
    raw_blog_post = response.content.strip()
    undesired_phrase = "Let me know if you have any questions!"
    while undesired_phrase in raw_blog_post:
        raw_blog_post = raw_blog_post.replace(undesired_phrase, "")
    return {"blog_post": raw_blog_post}

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
url = st.text_input("Enter YouTube Video URL", placeholder="e.g., https://youtu.be/your_video_id or youtube.com/shorts/...")

if st.button("Process Video"):
    if url:
        video_id = extract_video_id(url)
        
        if video_id:
            with st.spinner("Processing video: Fetching transcript, summarizing, and generating blog post..."):
                final_result = state_machine.invoke({"video_id": video_id})
                st.session_state.blog_post = final_result.get("blog_post")
                st.session_state.reviewed_summary = final_result.get("summary")
            st.success("Blog post generated!")
        else:
            st.error("Invalid YouTube URL. Please enter a valid video, share, or shorts link.")
    else:
        st.error("Please enter a YouTube URL.")

if "blog_post" in st.session_state:
    st.header("Your Generated Blog Post")
    st.markdown(st.session_state.blog_post)
