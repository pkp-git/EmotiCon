

import streamlit as st
import tempfile
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def show_upload_ui():
    st.title("Welcome to EmotiCon!")
    st.header("Step 1: Upload Audio File (WAV)")
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"], key="audio")
    st.header("Step 2: Upload Context File (PDF)")
    pdf_file = st.file_uploader("Choose a context PDF file", type=["pdf"], key="pdf")
    run_button = st.button("Run Analysis")
    if run_button and audio_file and pdf_file:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
            pdf_path = os.path.join(tmpdir, pdf_file.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.read())
            st.info("Running audio analysis (this may take a while)...")
            import subprocess
            segments_csv = os.path.join(data_dir, "segments.csv")
            segments_with_context_csv = os.path.join(data_dir, "segments_with_context.csv")
            audio_cmd = f'python audio.py "{audio_path}" "{segments_csv}"'
            import subprocess
            try:
                result = subprocess.run(audio_cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                st.error(f"audio.py failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
                return
            st.info("Running context analysis...")
            rag_cmd = f'python rag.py "{pdf_path}" "{segments_csv}" "{segments_with_context_csv}"'
            try:
                result = subprocess.run(rag_cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                st.error(f"rag.py failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
                return
            st.session_state.page = "results"
            st.success("Analysis complete! Redirecting to results...")
            st.rerun()
    else:
        st.info("Please upload both files and click 'Run Analysis'.")

def show_results_ui():
    st.title("EmotiCon: Emotion Analysis and Recommendations")
    # --- Load and check CSV ---
    data_dir = "data"
    segments_with_context_csv = os.path.join(data_dir, "segments_with_context.csv")
    if not os.path.exists(segments_with_context_csv):
        st.error("No results found. Please run analysis first.")
        if st.button("Back to Upload"):
            st.session_state.page = "upload"
        return
    df = pd.read_csv(segments_with_context_csv)
    # --- App logic from app.py ---
    EMOJI_MAP = {
        "anger": "😠", "ang": "😠",
        "joy": "😄", "hap": "😄",
        "sadness": "😢", "sad": "😢",
        "fear": "😨","sad": "😢",
        "surprise": "😲","hap": "😄",
        "disgust": "🤢","ang": "😠",
        "neutral": "😐", "neu": "😐"
    }
    def get_emoji(emotion):
        return EMOJI_MAP.get(str(emotion).lower().strip(), "")
    def normalize_emotion(emotion):
        e = str(emotion).lower().strip()
        if e in ["ang", "anger"]:
            return "ang"
        if e in ["hap", "joy", "happy"]:
            return "hap"
        if e in ["sad", "sadness"]:
            return "sad"
        if e in ["surprise", "fear", "disgust"]:
            return "neu"
        return "neu"
    def parse_emotion(emotion_str):
        if "(" in str(emotion_str):
            label, conf = emotion_str.split("(")
            label = label.strip()
            conf = conf.replace(")", "").strip()
            return label, conf
        return emotion_str, ""
    def emotion_match_color(audio_emotion, text_emotion, context_emotion):
        a, _ = parse_emotion(audio_emotion)
        t, _ = parse_emotion(text_emotion)
        c, _ = parse_emotion(context_emotion)
        a_norm = normalize_emotion(a)
        t_norm = normalize_emotion(t)
        c_norm = normalize_emotion(c)
        match_text = a_norm == t_norm
        match_context = a_norm == c_norm
        match_text_context = t_norm == c_norm
        if match_text and match_context and match_text_context:
            return "#2ECC40"
        elif match_text or match_context or match_text_context:
            return "#FFD700"
        else:
            return "#FF4136"
    # Sidebar segment selector
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0
    with st.sidebar:
        st.markdown("### Select Segment")
        selected_idx = st.selectbox(
            "Segment Number:",
            options=range(len(df)),
            index=st.session_state.selected_idx,
            key="segment_select",
            format_func=lambda i: f"Segment {i+1}: {df.iloc[i]['Segment']} ({df.iloc[i]['Audio Emotion']})"
        )
        if selected_idx != st.session_state.selected_idx:
            st.session_state.selected_idx = selected_idx
            st.rerun()
    row = df.iloc[st.session_state.selected_idx]
    # Plot segments as colored bars
    import plotly.graph_objects as go
    fig = go.Figure()
    for i, r in df.iterrows():
        color = emotion_match_color(r["Audio Emotion"], r["Text Emotion"], r["Context Emotion"])
        fig.add_trace(go.Bar(
            x=[float(r["Segment"].split("-")[1]) - float(r["Segment"].split("-")[0])],
            y=["Segment"],
            base=float(r["Segment"].split("-")[0]),
            orientation='h',
            marker_color=color,
            name=f"Segment {i+1}",
            hovertemplate=(
                f"<b>Segment:</b> {r['Segment']}<br>"
                f"<b>Audio Emotion:</b> {r['Audio Emotion']}<br>"
                f"<b>Text Emotion:</b> {r['Text Emotion']}<br>"
                f"<b>Context Emotion:</b> {r['Context Emotion']}<br>"
                f"<extra></extra>"
            ),
            showlegend=False
        ))
    fig.update_layout(
        barmode='stack',
        xaxis_title="Time (seconds)",
        yaxis=dict(showticklabels=False),
        height=180,
        margin=dict(l=20, r=20, t=20, b=20),
        title="Segments"
    )
    plot_col, legend_col = st.columns([6, 1])
    with plot_col:
        st.plotly_chart(fig, use_container_width=True)
    with legend_col:
        st.markdown("### Legend")
        st.markdown(
            "<div style='background-color: #2ECC40; padding: 5px; border-radius: 3px; text-align: center; color: white; font-weight: bold;'>🟩 All emotions match</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='background-color: #FFD700; padding: 5px; border-radius: 3px; text-align: center; color: black; font-weight: bold;'>🟨 Any two emotions match</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='background-color: #FF4136; padding: 5px; border-radius: 3px; text-align: center; color: white; font-weight: bold;'>🟥 No emotions match</div>",
            unsafe_allow_html=True
        )
    col1, col2, col3 = st.columns([1, 1, 2], gap="large")
    with col1:
        st.markdown("### 🎭 Emotions")
        audio_label, audio_conf = parse_emotion(row["Audio Emotion"])
        text_label, text_conf = parse_emotion(row["Text Emotion"])
        context_label, context_conf = parse_emotion(row["Context Emotion"])
        st.markdown(f"**🎵 Audio:** {get_emoji(audio_label)} {audio_label}")
        if audio_conf:
            st.markdown(f"*Confidence: {audio_conf}*")
        st.markdown(f"**📝 Text:** {get_emoji(text_label)} {text_label}")
        if text_conf:
            st.markdown(f"*Confidence: {text_conf}*")
        st.markdown(f"**🔄 Context:** {get_emoji(context_label)} {context_label}")
        if context_conf:
            st.markdown(f"*Confidence: {context_conf}*")
    with col2:
        st.markdown("### 📖 Content")
        st.markdown("**Transcript:**")
        st.markdown(f"*\"{row['Transcript']}\"*")
        st.markdown("**Context:**")
        context_text = row['Context']
        max_chars = 300
        if len(str(context_text)) > max_chars:
            if st.button("See more", key=f"see_more_{row['Segment']}"):
                st.markdown(f"*{context_text}*")
            else:
                st.markdown(f"*{context_text[:max_chars]}...*")
        else:
            st.markdown(f"*{context_text}*")
    row_idx = st.session_state.selected_idx
    if row_idx > 0:
        transcript1 = df.iloc[row_idx - 1]["Transcript"]
    else:
        transcript1 = ""
    if row_idx < len(df) - 1:
        transcript2 = df.iloc[row_idx + 1]["Transcript"]
    else:
        transcript2 = ""
    with col3:
        st.markdown("### Gemini Recommendation")
        try:
            import google.generativeai as genai
        except ImportError:
            st.error("google-generativeai package not installed. Please install it with 'pip install google-generativeai'.")
            return
        @st.cache_data(show_spinner=False)
        def get_gemini_recommendation(audio_emotion, text_emotion, context_emotion, transcript, transcript1, transcript2, context):
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return "[Gemini API key not set. Please set GEMINI_API_KEY in your environment.]"
            try:
                genai.configure(api_key=api_key)
                prompt = f"""
You are an expert in communication and emotional intelligence. Given the following:
- Audio Emotion: {audio_emotion}
- Text Emotion: {text_emotion}
- Context Emotion: {context_emotion}
- Transcript (previous): \"{transcript1}\"
- Transcript (current): \"{transcript}\"
- Transcript (next): \"{transcript2}\"
- Context: \"{context}\"

Analyze the above and recommend a specific change in the text or audio delivery that would improve the emotional impact or appropriateness of this segment. Be concise and actionable. Give the recommendation in second person, as if speaking directly to the person who created the segment.
"""
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                response = model.generate_content(prompt, generation_config={"temperature": 0.2})
                return response.text.strip()
            except Exception as e:
                return f"[Gemini error: {e}]"
        if st.button("Get Gemini Recommendation", key=f"gemini_btn_{row['Segment']}"):
            with st.spinner("Contacting Gemini..."):
                rec = get_gemini_recommendation(
                    row["Audio Emotion"], row["Text Emotion"], row["Context Emotion"],
                    row["Transcript"], transcript1, transcript2, row["Context"]
                )
                st.session_state[f"gemini_rec_{row['Segment']}"] = rec
        rec = st.session_state.get(f"gemini_rec_{row['Segment']}")
        if rec:
            st.success(rec)
        else:
            st.info("Click the button to get a Gemini recommendation for this segment.")
    st.markdown("---")
    st.markdown("""
    ### 💡 How to Use:
    - **Select** a segment from the sidebar to view its details.
    - **View** the detailed emotion analysis and content below the plot.
    - **Hover** over segments in the plot to see a quick preview of emotions.
    """)
    if st.button("Back to Upload"):
        st.session_state.page = "upload"

if "page" not in st.session_state:
    st.session_state.page = "upload"
if st.session_state.page == "upload":
    show_upload_ui()
elif st.session_state.page == "results":
    show_results_ui()
