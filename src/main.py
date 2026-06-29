import os
from gpt4all import GPT4All
import re
from src import pintora
from diffusers import AutoPipelineForText2Image

current_dir = os.path.dirname(os.path.abspath(__file__))
persona_path = os.path.join(current_dir, "persona.txt")
comandos_path = os.path.join(current_dir, "comandos.txt")


# Configuração de caminho
model_path = "C:/Users/joaot/.cache/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
gpt4all = GPT4All(model_path, device="cpu", n_threads=8, allow_download=False)
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cpu")

print(f"Threads ativas: {gpt4all.model.thread_count()}")


def carregar_txt(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Você é NoellE, uma assistente prestativa."


def gerar_resposta(user_input):
    if not os.path.exists(model_path):
        return "Erro: Modelo não encontrado."

    personality = carregar_txt(persona_path)
    comandos = carregar_txt(comandos_path)

    prompt = f"{personality}\n{comandos}\nUsuário: {user_input}\nNoellE:"

    # O "disfarce" enquanto ela não começa a falar:
    print("NoellE está processando...", end="\r", flush=True)

    # Dicionário para armazenar o estado dentro do callback
    contexto = {"primeiro_token": True, "texto_completo": ""}


    # Esta função será chamada para CADA token gerado pelo modelo
    def resposta_callback(token_id, token_string):
        if contexto["primeiro_token"]:
            print(" " * 30, end="\r")
            print("NoellE: ", end="", flush=True)
            contexto["primeiro_token"] = False

        # Se o modelo tentar pular linha, paramos imediatamente de forma segura
        if "\n" in token_string:
            return False

        contexto["texto_completo"] += token_string
        print(token_string, end="", flush=True)
        return True  # Diz ao modelo para continuar gerando

    # CORREÇÃO: Chamada direta do gpt4all.generate com o seu callback
    gpt4all.generate(
        prompt,
        callback=resposta_callback,
        n_predict=400,
        repeat_penalty=1.2,
        repeat_last_n=64,
        temp=0.7
    )

    print()  # Pula a linha no terminal assim que o callback para o modelo
    return contexto["texto_completo"].strip()


def comandos(resposta):
    for match in re.finditer(r'(/\w+)\s+"([^"]+)"', resposta):
        comandos, argumento = match.group(1), match.group(2)
        if "/image" in comandos:
            pintora.criar_imagem(pipe, argumento)

if __name__ == "__main__":
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            break
        resposta = gerar_resposta(user_input)
        comandos(resposta)