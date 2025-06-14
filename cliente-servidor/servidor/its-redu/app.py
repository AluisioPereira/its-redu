from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import google.generativeai as its_redu
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Configuração inicial
# chave_secreta = userdata.get('SECRET_KEY')
its_redu.configure(api_key="SECRET_KEY")

#Listagem de documentos com funcionalidades para o perfil de tutor

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

# Rotas Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    message = data['message']
    
    # Processar a mensagem e gerar uma resposta automática
    # Chame a função para obter o trecho de texto
    response = gerar_e_buscar_consulta(message, df, model)
    
    # Retorne o trecho de texto como resposta JSON
    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(debug=True)