import torch
import torch.nn as nn
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import webbrowser
import threading
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)
# 定義模型架構
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 載入 TfidfVectorizer
vectorizer = load("tfidf_vectorizer.pkl")
input_dim = vectorizer.max_features  
hidden_dim = 256  
output_dim = 7    

# 載入模型
model = MLP(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("mlp_weights.pth", map_location=torch.device('cpu')))
model.eval()

# 定義情緒標籤與表情符號
emotion_labels = {
    0: ("Anger", "😠"),
    1: ("Fear", "😨"),
    2: ("Joy", "😊"),
    3: ("Love", "❤️"),
    4: ("Neutral", "😐"),
    5: ("Sadness", "😢"),
    6: ("Surprise", "😲")
}
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # 移除標點、數字
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        user_input = request.form["text"]
        cleaned = preprocess_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray().astype('float32')
        input_tensor = torch.tensor(vector, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        label, emoji = emotion_labels.get(pred, ("Unknown", "❓"))
        result = f"{user_input} {emoji}"
    return render_template("index.html", result=result)

def open_browser():
    webbrowser.open_new("http://localhost:5000")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, use_reloader=False)
