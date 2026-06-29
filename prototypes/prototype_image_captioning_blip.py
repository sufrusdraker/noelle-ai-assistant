import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Inicialização do processador e do modelo BLIP da Salesforce
print("Carregando modelo de visão computacional (BLIP)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def analisar_imagem(caminho_imagem):
    """
    Processa uma imagem local e gera uma descrição textual automática (Image Captioning).
    """
    if not os.path.exists(caminho_imagem):
        print(f"\n[Aviso] Arquivo '{caminho_imagem}' não encontrado.")
        print("Por favor, coloque uma imagem com esse nome na pasta para testar o modelo.")
        return

    # Abre a imagem garantindo a conversão para o canal de cores RGB
    image = Image.open(caminho_imagem).convert("RGB")

    # Prepara os tensores de entrada para o modelo PyTorch
    inputs = processor(image, return_tensors="pt")

    print("Analisando elementos da imagem...")
    # Executa a inferência de geração de texto baseado na imagem
    out = model.generate(**inputs)

    # Decodifica os tokens gerados de volta para string legível
    caption = processor.decode(out[0], skip_special_tokens=True)

    print(f"\nDescrição gerada pela IA: {caption}")
    return caption


if __name__ == "__main__":
    # Define um nome de imagem genérico no mesmo diretório para testes
    current_dir = os.path.dirname(os.path.abspath(__file__))
    imagem_teste = os.path.join(current_dir, "input_image.jpg")

    analisar_imagem(imagem_teste)