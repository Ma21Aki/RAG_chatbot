# app.py の冒頭に追加（Flask起動前）
from dotenv import load_dotenv
load_dotenv()


# app.py
from flask import Flask, request, render_template_string
from utils.pdf_loader import extract_text_from_pdfs
from rag.indexer import build_faiss_index, model as embed_model
from rag.query import retrieve_and_answer

# 対象のPDFファイル
pdf_paths = [
    "data/20250606mt議事録.pdf",
    "data/20250609MT議事録.pdf",
    "data/20250613mt.pdf"
]

# PDFから全文テキストを抽出
print("🔍 PDF読み込み中...")
text = extract_text_from_pdfs(pdf_paths)

# FAISSインデックス構築
print("📦 インデックス構築中...")
index, chunks = build_faiss_index(text)

# Flaskアプリ作成
app = Flask(__name__)

# シンプルなHTMLフォーム
TEMPLATE = '''
<!doctype html>
<title>議事録質問応答</title>
<h2>PDF議事録から回答生成</h2>
<form method=post>
  <input name=query type=text style="width:400px;" placeholder="ここに質問を入力してください">
  <input type=submit value="質問する">
</form>
{% if answer %}
<h3>回答:</h3>
<p>{{ answer|safe }}</p>
{% endif %}
'''

@app.route("/", methods=["GET", "POST"])
def rag_interface():
    answer = ""
    if request.method == "POST":
        user_query = request.form["query"]
        answer = retrieve_and_answer(user_query, index, chunks, embed_model)
    return render_template_string(TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

    