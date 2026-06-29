import requests

# Endereço local do servidor da API (ajuste a porta se necessário)
URL_SERVIDOR = "http://localhost:5000/chat"


def enviar_requisicao():
    print("=== Cliente de Testes HTTP Inicializado ===")
    user_input = input("Digite uma mensagem para enviar ao servidor: ")

    # Se o usuário não digitar nada, cancela o envio
    if not user_input.strip():
        print("Mensagem vazia. Cancelando envio.")
        return

    # Payload (corpo da requisição em formato JSON)
    data = {
        "prompt": user_input
    }

    try:
        print(f"Enviando dados para {URL_SERVIDOR}...")
        # Faz a requisição POST síncrona
        response = requests.post(URL_SERVIDOR, json=data, timeout=10)

        # Verifica se o servidor respondeu com sucesso (Status 200 OK)
        if response.status_code == 200:
            resposta_json = response.json()
            # Captura a resposta de forma segura mesmo se a chave não existir
            resposta_ia = resposta_json.get("response", "Chave 'response' não encontrada no JSON.")
            print(f"\nResposta do Servidor:\n{resposta_ia}")
        else:
            print(f"\n[Erro] O servidor retornou o status HTTP: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("\n[Erro de Conexão] Não foi possível conectar ao servidor.")
        print("Certifique-se de que o script do servidor está rodando em 'localhost:5000'.")
    except Exception as e:
        print(f"\n[Erro Inesperado]: {e}")


if __name__ == "__main__":
    enviar_requisicao()