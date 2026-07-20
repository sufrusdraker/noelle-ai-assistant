import os
import sys
import re
from pathlib import Path
from gpt4all import GPT4All
from diffusers import AutoPipelineForText2Image
from src import pintora

# 1. Diretórios dinâmicos do Projeto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "locais" else CURRENT_DIR

# Busca persona.txt e comandos.txt na raiz do projeto (ou no mesmo diretório)
persona_path = PROJECT_ROOT / "persona.txt"
comandos_path = PROJECT_ROOT / "comandos.txt"

# 2. Caminho dinâmico para a pasta do GPT4All na HOME de qualquer usuário
# No Windows resolve para: C:\Users\<usuario_atual>\.cache\gpt4all\...
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
USER_HOME = Path.home()
model_path = USER_HOME / ".cache" / "gpt4all" / MODEL_NAME

# Se preferir usar variáveis de ambiente (útil para quem quer mudar a pasta do modelo):
# model_path = os.getenv("GPT4ALL_MODEL_PATH", USER_HOME / ".cache" / "gpt4all" / MODEL_NAME)

# Inicialização do GPT4All com suporte a download caso o modelo não exista
gpt4all = GPT4All(MODEL_NAME, device="cpu", n_threads=8, allow_download=True)

print(f"Threads ativas: {gpt4all.model.thread_count()}")

pipe = None

def obter_pipe():
    global pipe
    if pipe is None:
        print("\n[Sistema] Carregando SDXL-turbo para a memória RAM...")
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
        pipe.to("cpu")
    return pipe


def carregar_txt(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Você é NoellE, uma assistente prestativa."


def gerar_resposta(user_input):
    personality = carregar_txt(persona_path)
    comandos = carregar_txt(comandos_path)

    prompt = f"{personality}\n{comandos}\nUsuário: {user_input}\nNoellE:"

    # O "disfarce" enquanto ela não começa a falar:
    print("NoellE está processando...", end="\r", flush=True)

    # Dicionário para armazenar o estado dentro do callback
    contexto = {"primeiro_token": True, "texto_completo": ""}

    def resposta_callback(token_id, token_string):
        if contexto["primeiro_token"]:
            print(" " * 30, end="\r")
            print("NoellE: ", end="", flush=True)
            contexto["primeiro_token"] = False

        # Se o token contiver quebra de linha:
        if "\n" in token_string:
            token_limpo = token_string.replace("\n", "")

            if token_limpo:
                contexto["texto_completo"] += token_limpo
                print(token_limpo, end="", flush=True)

            return False

        contexto["texto_completo"] += token_string
        print(token_string, end="", flush=True)
        return True

    gpt4all.generate(
        prompt,
        callback=resposta_callback,
        n_predict=400,
        repeat_penalty=1.2,
        repeat_last_n=64,
        temp=0.7
    )

    print()
    return contexto["texto_completo"].strip()


def comandos(resposta):
    for match in re.finditer(r'(/\w+)\s+"([^"]+)"', resposta):
        cmd, argumento = match.group(1), match.group(2)
        if "/image" in cmd:
            pipeline = obter_pipe()
            pintora.criar_imagem(pipeline, argumento)

if __name__ == "__main__":
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            break
        resposta = gerar_resposta(user_input)
        comandos(resposta)