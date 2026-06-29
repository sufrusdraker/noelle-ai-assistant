import os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image

# Inicialização do pipeline do Stable Diffusion otimizado para CPU
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cpu")


def criar_imagem_com_callback(pipeline, prompt):
    """
    Gera uma imagem capturando e decodificando os estados latentes
    intermediários do VAE a cada passo da inferência.
    """
    imagens_intermediarias = []

    # Função callback que intercepta o fim de cada passo de difusão
    def callback_passo(pipeline_inst, step, timestep, latents_dict):
        latents = latents_dict["latents"]

        with torch.no_grad():
            # Desnormalização matemática dos latentes do Stable Diffusion
            latents_dec = 1 / 0.18215 * latents

            # Decodificação manual utilizando o VAE (Variational Autoencoder) do modelo
            decoded = pipeline_inst.vae.decode(latents_dec).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()

            # Converte a matriz NumPy em uma imagem PIL legível
            pil_images = [Image.fromarray((im * 255).astype(np.uint8)) for im in decoded]

            # Salva o progresso visual do passo atual
            nome_arquivo = f"step_{step}.png"
            pil_images[0].save(nome_arquivo)
            imagens_intermediarias.append(pil_images[0])
            print(f" -> Passo {step} renderizado e salvo como '{nome_arquivo}'", flush=True)

        # Retorna o dicionário obrigatório para o pipeline continuar a execução
        return {"latents": latents}

    # Executa a inferência injetando o callback customizado
    pipeline(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=5.0,
        callback_on_step_end=callback_passo
    )


def main():
    print("Módulo experimental de interceptação de latentes inicializado.")
    while True:
        user_input = input("\nDesenhe a sua imaginação (ou digite 'sair'): ")
        if user_input.lower() in ["sair", "exit"]:
            print("Encerrando experimento.")
            break

        if user_input.strip():
            print("Iniciando geração com interceptação de passos...")
            criar_imagem_com_callback(pipe, user_input)
            print(f'Processo concluído para o prompt: "{user_input}"')


if __name__ == "__main__":
    main()