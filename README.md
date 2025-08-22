# EmotiCon – AI-Powered Emotion Analysis & Recommendations

Welcome to **EmotiCon**, an AI-driven system for analyzing emotions in audio and text, enriched with context and actionable recommendations! Whether you're interested in emotion recognition, context-aware analysis, or AI-powered feedback, EmotiCon brings it all together in a seamless workflow.

---

## Features

✅ **Multimodal Emotion Analysis** – Detect emotions from both audio and transcript using state-of-the-art AI models!  
✅ **Context Enrichment** – Add relevant context from PDF documents to each segment for deeper insights!  
✅ **AI Recommendations** – Get actionable, segment-specific feedback powered by Google Gemini!  
✅ **Interactive Streamlit UI** – Upload, analyze, and explore results in your browser!  
✅ **Seamless Workflow** – One-click analysis from upload to results!  

---

## How To Install?

### 1⃣ Clone the Repository
```bash
# or PowerShell
git clone https://github.com/pkp-git/EmotiCon.git
cd EmotiCon
```

### 2⃣ Set Up Environment Variables
- Create a `.env` file in the root directory (see the included example).
- Add your API keys:
  - `GEMINI_API_KEY` (Google Gemini)
  - `ASSEMBLYAI_API_KEY` (AssemblyAI)

### 3⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4⃣ Run the Application
```bash
streamlit run app.py
```

### 5⃣ Open in Browser
Go to **http://localhost:8501** and explore emotion analysis and recommendations! 🎭

---

## How It Works

1⃣ Upload an audio file (WAV/MP3) and a context PDF.  
2⃣ The system segments the audio, transcribes, and analyzes emotions.  
3⃣ Context is retrieved and matched to each segment using semantic search.  
4⃣ The results UI visualizes segments, emotions, and context.  
5⃣ Get AI-powered recommendations for each segment with a click!

---

## Contributing

💡 Have ideas or improvements? Found a bug? Want to enhance EmotiCon?  
Fork the repo, make your changes, and submit a **pull request**!

---

## 📜 License

📝 This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

Special thanks to the open-source community and all contributors who made this project possible!
