import os
from diffusers import AutoPipelineForText2Image

# Configuração do Pipeline do Stable Diffusion (SDXL Turbo rodando em CPU)
# O modelo será compartilhado se este módulo for importado por outro script
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cpu")


def criar_imagem(pipeline, prompt, output_name="output.png"):
    """
    Gera uma imagem a partir de um prompt de texto e a salva localmente.
    """
    # Executa a inferência com poucas etapas (otimizado para a velocidade do sdxl-turbo)
    resultado = pipeline(prompt=prompt, num_inference_steps=4, guidance_scale=1.5)
    image = resultado.images[0]

    image.show()  # Abre o visualizador padrão do sistema
    image.save(output_name)  # Salva o arquivo na raiz de execução
    return output_name


def main():
    print("Módulo de geração de imagens inicializado.")
    while True:
        user_input = input("Desenhe a sua imaginação (ou digite 'sair'): ")

        if user_input.lower() in ["sair", "exit"]:
            print("Encerrando módulo de pintura.")
            break

        if user_input.strip():
            print("Processando imagem... Isso pode levar alguns minutos na CPU.")
            criar_imagem(pipe, user_input)
            print(f'Imagem gerada com sucesso para o prompt: "{user_input}"\n')


if __name__ == "__main__":
    main()