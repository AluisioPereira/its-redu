from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import google.generativeai as its_redu
import json
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuração inicial [colocar a chave aqui]
its_redu.configure(api_key="CHAVE_API")

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
model = "models/embedding-001"

# Função para embeddar o texto
def embed_fn(title, text):
    return its_redu.embed_content(model=model,
                                  content=text,
                                  title=title,
                                  task_type="RETRIEVAL_DOCUMENT")["embedding"]

df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)

# Função para gerar e buscar consulta
def gerar_e_buscar_consulta(message, df, model):
    embedding_da_consulta = its_redu.embed_content(model=model,
                                 content=message,
                                 task_type="RETRIEVAL_QUERY")["embedding"]

    produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

    indice = np.argmax(produtos_escalares)
    return df.iloc[indice]["Conteudo"]

# Chat com Embeddings
# Rotas Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    response = gerar_e_buscar_consulta(message, df, model)
    
    return jsonify({'message': response})

# - CHAT NORMAL
# Inicializando o modelo generativo
generation_config = {
  "candidate_count": 1,
  "temperature": 0.5,
}

safety_settings = {
    'HATE': True,
    'HARASSMENT': True,
    'SEXUAL': True,
    'DANGEROUS': True,
}

model_generative = its_redu.GenerativeModel(model_name='gemini-1.0-pro',
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

# Função para enviar mensagem no chat normal
@app.route('/send_message_chat', methods=['POST'])
def send_message_chat():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    response = model_generative.generate_content(message)

    return jsonify({'message': response.text})

# WhatsApp
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
        
        # Obter a resposta do modelo Gemini
        response_text = gerar_e_buscar_consulta(message, df, model)
        
        # Enviar a resposta de volta para o WhatsApp
        send_whatsapp_message(from_number, response_text)
        
    return jsonify({'status': 'success'})


@app.route('/send_message_incorporacao', methods=['POST'])
def send_message_incorporacao():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    response = gerar_e_buscar_consulta(message, df, model)    
    # Processar a mensagem e gerar uma resposta automática
    response_incorp = model_generative.generate_content("Enriqueça este texto: "+ response)

    return jsonify({'message': response_incorp.text})


if __name__ == '__main__':
    app.run(debug=True)
