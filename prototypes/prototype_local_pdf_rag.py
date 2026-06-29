import os
from gpt4all import GPT4All
from pypdf import PdfReader

# Configuração dinâmica do caminho do modelo local
home_dir = os.path.expanduser("~")
model_path = os.path.join(home_dir, ".cache", "gpt4all", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")

print("Carregando modelo LLM para análise de documentos...")
model = GPT4All(model_path)


def ler_pdf(caminho_pdf):
    """Extrai todo o texto legível de um arquivo PDF página por página."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado em: {caminho_pdf}")

    reader = PdfReader(caminho_pdf)
    texto = ""
    for pagina in reader.pages:
        texto_extraido = pagina.extract_text()
        if texto_extraido:
            texto += texto_extraido + "\n"
    return texto


def dividir_texto(texto, tamanho_bloco=800):
    """Quebra strings longas em pequenos blocos (chunks) para caber no contexto da IA."""
    return [texto[i:i + tamanho_bloco] for i in range(0, len(texto), tamanho_bloco)]


def buscar_contexto(pergunta, chunks):
    """Mecanismo simplificado de busca (Keyword Matching RAG)."""
    relevantes = []
    palavras = pergunta.lower().split()

    for chunk in chunks:
        # Se qualquer palavra da pergunta estiver no bloco, considera relevante
        if any(p in chunk.lower() for p in palavras):
            relevantes.append(chunk)

    # Limita o retorno aos 3 blocos mais relevantes para não estourar o contexto
    return "\n".join(relevantes[:3])


def gerar_resposta(user_input, chunks):
    """Injeta as regras da persona e o contexto recuperado do PDF no prompt final."""
    personality = (
        "[SYSTEM PROMPT]\n"
        "Você é NoellE, uma inteligência artificial prestativa e analítica.\n"
        "Responda a pergunta do usuário baseando-se estritamente no contexto do documento fornecido abaixo."
    )

    contexto = buscar_contexto(user_input, chunks)

    prompt = f"""{personality}

[CONTEXTO EXTRAÍDO DO DOCUMENTO]
{contexto}

[PERGUNTA DO USUÁRIO]
{user_input}

NoellE responde:"""

    response = model.generate(prompt, n_predict=200)
    return response.split("\n")[0].strip()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define um nome de arquivo genérico na mesma pasta para testes
    pdf_teste = os.path.join(current_dir, "documento_teste.pdf")

    print(f"\n--- Sistema Local de RAG Inicializado ---")
    print(f"Para testar, coloque um arquivo PDF renomeado para 'documento_teste.pdf' em: {current_dir}")

    try:
        texto_completo = ler_pdf(pdf_teste)
        blocos_de_texto = dividir_texto(texto_completo)

        pergunta = "Qual o tema principal abordado no texto?"
        print(f"\nProcessando pergunta: '{pergunta}'...")

        resposta = gerar_resposta(pergunta, blocos_de_texto)
        print(f"\nResposta da IA:\n{resposta}")

    except FileNotFoundError as e:
        print(f"\n[Aviso] {e}")