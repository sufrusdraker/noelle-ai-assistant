import os
import json
import re
from gpt4all import GPT4All
from flask import Flask, request, jsonify

# Configuração dinâmica de caminhos (independente de sistema operacional)
current_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.expanduser("~")
model_path = os.path.join(home_dir, ".cache", "gpt4all", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
memory_file = os.path.join(current_dir, "memoria_api.json")

# Inicialização do Flask e do modelo local
app = Flask(__name__)
print("Carregando modelo LLM para a API local...")
gpt4all = GPT4All(model_path)


def carregar_historico(quantidade=5):
    """Carrega as últimas N interações do arquivo JSON de forma segura."""
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            try:
                historico = json.load(f)
                if not isinstance(historico, list):
                    raise ValueError("Formato inválido.")
            except (json.JSONDecodeError, ValueError):
                print("[Aviso] Arquivo de memória inválido. Resetando histórico.")
                historico = []
        return historico[-quantidade:]
    return []


def salvar_interacao(user_input, response):
    """Registra o novo par de perguntas/respostas no banco de dados local."""
    interacao = {"usuario": user_input, "noelle": response}
    # Carrega todo o histórico para não sobrescrever dados antigos
    historico_completo = carregar_historico(quantidade=100)
    historico_completo.append(interacao)
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(historico_completo, f, indent=4, ensure_ascii=False)


@app.route('/chat', methods=['POST'])
def gerar_resposta():
    if not os.path.exists(model_path):
        return jsonify({"error": "Modelo GGUF não encontrado no caminho especificado."}), 500

    data = request.get_json() or {}
    user_input = data.get("prompt", "").strip()

    if not user_input:
        return jsonify({"error": "O campo 'prompt' não pode estar vazio."}), 400

    # Diretrizes de comportamento do modelo
    personality = (
        "[SYSTEM PROMPT]\n"
        "Você é NoellE, uma inteligência artificial local com personalidade própria.\n"
        "Responda de forma clara, objetiva, organizada e em português (pt-BR)."
    )

    # Resgata o contexto das rodadas passadas
    historico = carregar_historico()
    contexto = "\n".join([f"Usuário: {h['usuario']}\nNoellE: {h['noelle']}" for h in historico])

    # Montagem final do prompt estruturado
    prompt = f"{personality}\n\n[CONTEXTO DA CONVERSA]\n{contexto}\n\nUsuário: {user_input}\nNoellE:"

    # Processamento e pós-tratamento da string gerada
    response = gpt4all.generate(prompt, n_predict=150)
    response = response.split("\n")[0].strip()
    response = re.sub(r'\*.*?\*', '', response)  # Remove ações entre asteriscos
    response = re.sub(r'\s+', ' ', response).strip()  # Remove espaços duplicados

    # Armazena a rodada na memória local
    salvar_interacao(user_input, response)

    return jsonify({"response": response})


if __name__ == '__main__':
    print("\n Servidor local rodando! Aguardando requisições em http://localhost:5000/chat")
    # Rodar em modo de produção simulado sem o autoreload do debug para evitar travar o modelo
    app.run(host="127.0.0.1", port=5000, debug=False)