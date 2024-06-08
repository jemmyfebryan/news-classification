import os

from flask import Flask, request, render_template
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from embedder import get_sentence_embedding
from model.assembly import reassemble_model
from model.news_model import BiLSTM_Attention
import model.config as cfg
from prepro import all_prepro

app = Flask(__name__)

model_dir = 'model/indobert_model'
output_file = os.path.join(model_dir, 'model.safetensors')
part_prefix = os.path.join(model_dir, 'model_part_a')
num_parts = len([f for f in os.listdir(model_dir) if f.startswith('model_part_a')])

reassemble_model(output_file, part_prefix, num_parts)

tokenizer = AutoTokenizer.from_pretrained("model/indobert_tokenizer")
bert_model = AutoModel.from_pretrained("model/indobert_model")

news_model = BiLSTM_Attention(
    input_size=cfg.news_model.input_size,
    hidden_size=cfg.news_model.hidden_size,
    num_layers=cfg.news_model.num_layers,
    num_classes=cfg.news_model.num_classes
)
news_model.load_state_dict(torch.load('model/news_model.pth'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            text_input = request.form.get('text_input')
            text_prepro = all_prepro(text_input)
            text_embedding = get_sentence_embedding(
                tokenizer=tokenizer,
                bert_model=bert_model,
                text=text_prepro,
                max_sequence_length=cfg.BERT.max_sequence_length
            )
            text_tensor = torch.tensor(np.array([text_embedding]), dtype=torch.float32)
            cfg.news_model.set_seed(42)
            news_model.eval()
            with torch.no_grad():
                predict_result = news_model(text_tensor)

            predict_class = np.round(predict_result[0]).numpy()
            predict_class_text = [
                cfg.news_model.output_labels[j] for j in range(len(cfg.news_model.output_labels)) if predict_class[j] == 1
            ]

            response = {
                "result": predict_class_text,
                "status": {
                    "code": 200,
                    "message": "news predicted successfully"
                }
            }
            return response
        except Exception as e:
            response = {
                "result": None,
                "status": {
                    "code": 500,
                    "message": str(e)
                }
            }
            return response

if __name__ == '__main__':
    app.run(debug=False)
