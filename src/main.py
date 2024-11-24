import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Carregar o modelo SAM
model_type = "vit_h"  # Escolha entre "vit_h", "vit_l" ou "vit_b"
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"  # Caminho para o checkpoint do modelo SAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Registrar e carregar o modelo SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Diretório de saída para salvar as máscaras
output_dir = "./data/generated"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    # Carregar a imagem
    image_folder_path = "./data/samples"
    for image_name in os.listdir(image_folder_path):
        image = cv2.imread(os.path.join(image_folder_path,image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter BGR para RGB

        # Gerar máscaras automaticamente
        masks = mask_generator.generate(image)
        print(f"{len(masks)} máscaras geradas.")

        # # Salvar máscaras como imagens binárias
        # for i, mask in enumerate(masks):
        #     # Máscara binária (0 ou 1)
        #     binary_mask = (mask["segmentation"] * 255).astype("uint8")  # Converter para escala de 0-255

        #     # Caminho para salvar a máscara
        #     mask_path = os.path.join(output_dir, f"mask_{i}.png")
        #     cv2.imwrite(mask_path, binary_mask)
        #     print(f"Máscara salva em: {mask_path}")

        # Criar sobreposição de máscaras na imagem original
        overlay_image = image.copy()

        # Atribuir uma cor para cada máscara
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Vermelho, Verde, Azul, Amarelo
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]  # Escolher cor cíclica
            segmentation = mask["segmentation"]

            # Aplicar cor sobre a área da máscara
            overlay_image[segmentation] = overlay_image[segmentation] * 0.5 + np.array(color) * 0.5

        # Salvar a imagem com as máscaras sobrepostas
        overlay_path = os.path.join(output_dir, f"overlay_image_{image_name}")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))  # Converter para BGR antes de salvar
        print(f"Imagem com sobreposição salva em: {overlay_path}")

        # Exibir a imagem sobreposta
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.title("Imagem com Máscaras Sobrepostas")
        plt.axis("off")
        plt.show()
