from flask import Flask, request, jsonify, render_template_string
import os
import threading
import webbrowser
import rag_pipeline

# Initialize Flask
app = Flask(__name__)

# Application State
embedding_model = None
llm_pipeline = None
chunks = []
chunk_embeddings = None
filepath = "document.txt"

def load_models_and_data():
    global embedding_model, llm_pipeline, chunks, chunk_embeddings
    print("[INIT] Loading Document...")
    # Ensuring the initial document exists (same logic as the CLI)
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("Retrieval-Augmented Generation (RAG) is a technique that enhances generative AI models with facts fetched from external sources. ")
            f.write("The primary advantage of RAG is that it grounds the model on truth, reducing hallucinations. ")
            f.write("Sentence Transformers are lightweight models used to convert text into vector embeddings. ")
            f.write("A vector embedding is an array of numbers representing the semantic meaning of text. ")
            f.write("Cosine similarity is an algorithm used to measure how similar two vectors are.")
            
    chunks = rag_pipeline.load_and_split_document(filepath)
    
    print("[INIT] Loading Sentence Transformer Embedding Model...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = embedding_model.encode(chunks)
    
    print("[INIT] Loading Local LLM Pipeline (google/flan-t5-small)...")
    from transformers import pipeline
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
    print("[READY] Server is fully loaded and ready to accept queries!")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural RAG Engine | Local Edition</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f111a;
            --container-bg: rgba(25, 28, 41, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --primary: #4F46E5;
            --primary-glow: rgba(79, 70, 229, 0.5);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-main);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }
        .container {
            width: 100%;
            max-width: 800px;
            background: var(--container-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            height: 85vh;
        }
        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            margin-top: 0;
            margin-bottom: 24px;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            flex-shrink: 0;
        }
        .chat-box {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 30px;
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 10px;
        }
        .message {
            padding: 16px 20px;
            border-radius: 16px;
            max-width: 85%;
            animation: fadeIn 0.4s ease-out;
            line-height: 1.6;
        }
        .message.user {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .message.bot {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.2), rgba(192, 132, 252, 0.1));
            border: 1px solid rgba(79, 70, 229, 0.3);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .context-badge {
            display: inline-block;
            font-size: 0.75rem;
            background: rgba(0,0,0,0.3);
            padding: 4px 10px;
            border-radius: 12px;
            margin-top: 12px;
            color: var(--text-muted);
            border: 1px solid rgba(255,255,255,0.05);
            cursor: help;
        }
        .input-area {
            display: flex;
            gap: 12px;
            position: relative;
            flex-shrink: 0;
        }
        input {
            flex: 1;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            padding: 16px 24px;
            border-radius: 999px;
            color: white;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            outline: none;
            transition: all 0.3s ease;
        }
        input:focus {
            border-color: #818cf8;
            box-shadow: 0 0 15px rgba(129, 140, 248, 0.2);
        }
        button {
            background: linear-gradient(135deg, #4F46E5, #9333ea);
            border: none;
            color: white;
            padding: 0 32px;
            border-radius: 999px;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px var(--primary-glow);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--primary-glow);
        }
        button:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .loading-dots:after {
            content: '.';
            animation: dots 1.5s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Neural RAG Engine</h1>
        <div class="chat-box" id="chat">
            <div class="message bot">
                Greetings! I am currently analyzing your local document (`document.txt`). Ask me any question based on its contents, and I'll retrieve the context and formulate an answer just for you.
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="queryInput" placeholder="Ask a question about the document..." autocomplete="off" onkeypress="if(event.key === 'Enter') submitQuery()">
            <button id="sendBtn" onclick="submitQuery()">Synthesize</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat');
        const input = document.getElementById('queryInput');
        const btn = document.getElementById('sendBtn');

        function addMessage(text, sender, context = null) {
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.innerHTML = text.replace(/\\n/g, '<br>');
            
            if (context && context.length > 0) {
                const badge = document.createElement('div');
                badge.className = 'context-badge';
                badge.innerText = `Synthesized from ${context.length} relevant chunks`;
                badge.title = 'Top Context Snippets:\n\\n' + context.join('\\n\\n');
                div.appendChild(document.createElement('br'));
                div.appendChild(badge);
            }
            
            chatBox.appendChild(div);
            // Wait for DOM
            setTimeout(() => chatBox.scrollTop = chatBox.scrollHeight, 10);
            return div;
        }

        async function submitQuery() {
            const query = input.value.trim();
            if(!query) return;

            input.value = '';
            input.disabled = true;
            btn.disabled = true;
            addMessage(query, 'user');
            
            const loadingMsg = addMessage("Retrieving context & generating answer<span class='loading-dots'></span>", 'bot');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query})
                });
                
                const data = await response.json();
                chatBox.removeChild(loadingMsg);

                if (response.ok) {
                    addMessage(data.answer, 'bot', data.context);
                } else {
                    addMessage("<b>System Error:</b> Backend is still initializing, please wait a moment.", 'bot');
                }
                
            } catch (error) {
                if (chatBox.contains(loadingMsg)) {
                    chatBox.removeChild(loadingMsg);
                }
                addMessage("<b>Error:</b> Neural synthesis failed. Check console.", 'bot');
            }

            input.disabled = false;
            btn.disabled = false;
            input.focus();
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ask", methods=["POST"])
def ask():
    if llm_pipeline is None or embedding_model is None:
        return jsonify({"error": "Models are still initializing in the background. Please wait ~15 seconds and try again."}), 503

    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    # 1. Embed query
    query_embedding = embedding_model.encode(query)
    
    # 2. Retrieve top chunks
    k = min(3, len(chunks))
    top_results = rag_pipeline.retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=k)
    
    # 3. Generate Answer
    answer = rag_pipeline.generate_answer(query, top_results, llm_pipeline)
    
    # Formatting output correctly for the UI
    context_strings = [res["chunk"] for res in top_results]
    
    return jsonify({
        "answer": answer,
        "context": context_strings
    })

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # Start eager loading of the Machine Learning models in a background thread
    # so the web server starts instantly and users don't think it hung
    loading_thread = threading.Thread(target=load_models_and_data)
    loading_thread.start()
    
    print("Starting Premium Web Interface at http://127.0.0.1:5000/")
    # Automatically open the browser tab
    threading.Timer(1.5, open_browser).start()
    
    # Run the server
    app.run(host="127.0.0.1", port=5000, debug=False)
