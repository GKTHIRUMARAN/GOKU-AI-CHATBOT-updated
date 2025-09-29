# Goku Chatbot – Personalized Conversational AI

A personalized AI chatbot designed to talk like Son Goku (Dragon Ball).
This project uses Meta LLaMA 3 8B Instruct (via LM Studio) combined with Gradio for deployment.
It supports persona prompting, memory persistence, and knowledge augmentation (RAG) for authentic, character-driven conversations.

#🚀 Features
⚡ Goku Persona – Chatbot speaks and responds like Goku.
🧠 Memory System – Stores past conversations in memory.txt for continuity.
📚 Knowledge Injection – Uses knowledge.txt (Dragon Ball wiki excerpts) for lore accuracy.
🎨 Gradio Interface – Clean, simple UI for local or public deployment.
🔧 Modular Design – Easy to extend with new models, RAG pipelines, or character personas.

#🏗️ Project Structure
GokuChatbot/
│── app.py              # Main Gradio app
│── prompt.txt          # Persona definition for Goku
│── knowledge.txt       # Knowledge base (Dragon Ball wiki data)
│── memory.txt          # Stores chat history
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation

#🔑 Requirements

Python 3.9+

LM Studio
 (running Meta LLaMA 3 8B Instruct)

Packages from requirements.txt:

gradio
requests


(add langchain, faiss, etc. if extending with RAG)

#▶️ Run Locally

Start LM Studio with Meta LLaMA 3 8B Instruct loaded.

Make sure it is listening on http://localhost:1234/v1 (default).

Clone the repo:

git clone https://github.com/your-username/GokuChatbot.git
cd GokuChatbot


Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


Open your browser → http://localhost:7860

#⚙️ How It Works

Persona Prompting – Reads prompt.txt to enforce Goku’s character style.

Memory – Appends every user–bot exchange to memory.txt.

Knowledge – Uses knowledge.txt for lore-accurate answers (future: RAG retrieval).

Deployment – Served via Gradio (local + optional public link).

#📌 Roadmap

 Improve RAG with FAISS/Chroma vector store for knowledge search.

 Add streaming responses for faster chat.

 Multi-character support (Vegeta, Piccolo, etc.).

 Dockerize for portable deployment.

 WebSocket API for integration into apps/games.

#📜 License
MIT License – free to use, modify, and distribute.

#👤 Author
GK Thirumaran\
🎓 B.Tech Artificial Intelligence and Data Science\
🌍 Coimbatore, Tamil Nadu, India\
💼 Aspiring Data Scientist & Analyst | AIML Developer\
🔗 [Linkedin](https://www.linkedin.com/in/thirumarangk-ai) | [Porfolio](https://maranthiru180.wixsite.com/my-site)
