from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuração do modelo LLaMA
model_name = "meta-llama/LLaMA-7b"  # Certifique-se de substituir pelo modelo exato que está utilizando
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Função para carregar os documentos do arquivo JSON
def load_documents_from_json(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        documents = json.load(file)
    return documents

# Obtendo o diretório atual do arquivo Python
current_directory = os.path.dirname(os.path.abspath(__file__))

# Definindo o caminho absoluto para o arquivo JSON
tutor_json = os.path.join(current_directory, "db", "tutor.json")

# Carregar os documentos do arquivo JSON
tutor_documents = load_documents_from_json(tutor_json)

df = pd.DataFrame(tutor_documents)
df.columns = ["Titulo", "Conteudo"]

# Placeholder para embeddings (use um método adequado se necessário)
df["Embeddings"] = [np.random.rand(768) for _ in range(len(df))]  # Exemplo: Random embeddings

# Função para gerar e buscar consulta (adapte para sua lógica específica de embeddings)
def gerar_e_buscar_consulta(message, df):
    # Gere um embedding de consulta simuladamente ou use uma biblioteca de embeddings
    embedding_da_consulta = np.random.rand(768)  # Substitua com sua lógica de embedding

    # Calcule similaridade usando produto escalar
    produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)
    indice = np.argmax(produtos_escalares)
    return df.iloc[indice]["Conteudo"]

# Função para gerar respostas usando o LLaMA
def gerar_resposta_llama(message):
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100, temperature=0.5)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Rotas Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    response = gerar_e_buscar_consulta(message, df)
    
    return jsonify({'message': response})

# Função para enviar mensagem no chat normal
@app.route('/send_message_chat', methods=['POST'])
def send_message_chat():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    response = gerar_resposta_llama(message)

    return jsonify({'message': response})

# Configurações da API do WhatsApp
WHATSAPP_API_URL = 'https://api.whatsapp.com/send'
WHATSAPP_API_TOKEN = 'seu-token-de-autorizacao'

def send_whatsapp_message(to, message):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {WHATSAPP_API_TOKEN}',
    }
    data = {
        'to': to,
        'type': 'text',
        'text': {
            'body': message
        }
    }
    response = requests.post(WHATSAPP_API_URL, headers=headers, json=data)
    return response.json()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if 'messages' in data:
        message = data['messages'][0]['text']
        from_number = data['messages'][0]['from']
        
        # Obter a resposta usando a lógica de busca e consulta
        response_text = gerar_e_buscar_consulta(message, df)
        
        # Enviar a resposta de volta para o WhatsApp
        send_whatsapp_message(from_number, response_text)
        
    return jsonify({'status': 'success'})

@app.route('/send_message_incorporacao', methods=['POST'])
def send_message_incorporacao():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    response = gerar_e_buscar_consulta(message, df)    
    # Processar a mensagem e gerar uma resposta automática
    response_incorp = gerar_resposta_llama("Enriqueça este texto: " + response)

    return jsonify({'message': response_incorp})

if __name__ == '__main__':
    app.run(debug=True)
