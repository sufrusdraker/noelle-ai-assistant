import os
from gpt4all import GPT4All
# Importa a classe de emoções que refatoramos
from emotions import EmotionalEngine

# Configuração dinâmica de caminhos (funciona em qualquer sistema operacional)
current_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.expanduser("~")
model_path = os.path.join(home_dir, ".cache", "gpt4all", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
persona_path = os.path.join(current_dir, "persona.txt")

# Inicialização do modelo local na CPU
gpt4all = GPT4All(model_path, device="cpu", n_threads=8, allow_download=False)
# Inicialização do motor de estados emocionais
emotional_engine = EmotionalEngine()

print(f"Threads ativas no modelo: {gpt4all.model.thread_count()}")


def carregar_persona():
    """Carrega as diretrizes de personalidade do arquivo de texto."""
    if os.path.exists(persona_path):
        with open(persona_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Você é NoellE, uma assistente prestativa."


def gerar_resposta(user_input):
    if not os.path.exists(model_path):
        return "Erro: Modelo não encontrado."

    # Calcula o impacto emocional da frase do usuário
    conexao, postura, estabilidade = emotional_engine.analisar_impacto_emocional(user_input)

    persona_base = carregar_persona()

    # Injeção dinâmica do estado emocional no prompt do sistema
    prompt = (
        "[INSTRUÇÕES DE PERSONALIDADE - NÃO REPETIR]\n"
        f"{persona_base}\n\n"
        "[ESTADO EMOCIONAL DA NOELLE NESTE INSTANTE]\n"
        f"- Conexão com o Mestre: {conexao}/10 (Valores altos indicam que ela se sente amada; valores negativos indicam rejeição).\n"
        f"- Postura Defensiva/Orgulho: {postura}/10 (Valores altos significam que ela está muito orgulhosa/Tsundere; valores negativos significam guarda baixa).\n"
        f"- Estabilidade Emocional: {estabilidade}/10 (Valores negativos indicam que ela está nervosa, tímida ou estressada).\n\n"
        "[DIÁLOGO]\n"
        f"Usuário: {user_input}\n"
        "NoellE:"
    )

    print("NoellE está processando...", end="\r", flush=True)

    # Estado compartilhado para o controle de tokens no terminal
    contexto = {"primeiro_token": True, "texto_completo": ""}

    def resposta_callback(token_id, token_string):
        if contexto["primeiro_token"]:
            print(" " * 30, end="\r")  # Limpa o texto de processamento
            print("NoellE: ", end="", flush=True)
            contexto["primeiro_token"] = False

        # Trava de segurança para impedir quebras de linha indesejadas
        if "\n" in token_string:
            return False

        contexto["texto_completo"] += token_string
        print(token_string, end="", flush=True)
        return True

    # Execução da inferência via streaming
    gpt4all.generate(
        prompt,
        callback=resposta_callback,
        n_predict=400,
        repeat_penalty=1.2,
        repeat_last_n=64,
        temp=0.7
    )

    print()  # Pula linha ao encerrar a resposta
    return contexto["texto_completo"].strip()


if __name__ == "__main__":
    print("Iniciando NoellE v3 (Inteligência Artificial Local com Sistema Emocional Interativo)...")
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            print("Encerrando sessão.")
            break
        if user_input.strip():
            gerar_resposta(user_input)