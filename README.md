![Image](https://github.com/user-attachments/assets/43cc172d-0a65-4332-80c8-7fe714417535)



Este projeto visa desenvolver uma solução de Inteligência Artificial (IA) para o monitoramento de plantações, com foco na **identificação de pragas e doenças nas plantas** utilizando a combinação de **sensores IoT** e **machine learning**. A solução será baseada em **PyTorch** e usa **sensores IoT** para coletar dados ambientais que afetam a saúde das plantas.

## Tecnologias Utilizadas

- **PyTorch**: Framework de deep learning utilizado para construir e treinar modelos de aprendizado de máquina.
- **Sensores IoT**: Sensores para medir dados como temperatura, umidade, luminosidade e pH do solo.
- **Machine Learning**: Uso de redes neurais convolucionais (CNNs) para identificar padrões de doenças e pragas a partir de imagens de folhas e plantas.

## Funcionalidades

- Monitoramento de umidade do solo, temperatura e outros parâmetros ambientais para detectar condições favoráveis ao desenvolvimento de doenças.
- Identificação de pragas e doenças em plantas através de imagens capturadas por câmeras ou drones, utilizando redes neurais convolucionais.
- Armazenamento e análise de dados em tempo real.

## Como Começar

### 1. Instalar as Dependências

Clone o repositório e instale as dependências necessárias:

```
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
pip install -r requirements.txt

```

### 2. Coleta e Pré-processamento de Dados
O dataset será composto por imagens de folhas de plantas, coletadas com o auxílio de câmeras ou drones. O pré-processamento das imagens inclui redimensionamento, normalização e preparação dos dados para serem alimentados no modelo de IA.

### 3. Exemplo de Código para Pré-processamento de Imagens

```

import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('caminho/para/dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


```

### 4. Arquitetura do Modelo
O modelo de IA utilizado é uma rede neural convolucional (CNN), que é ideal para análise de imagens.

```

import torch.nn as nn
import torch.optim as optim

class PlantDiseaseCNN(nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*56*56, 128)
        self.fc2 = nn.Linear(128, 2)  # Ajuste conforme o número de classes (doença ou não)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*56*56)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

### 5. Treinamento do Modelo
Após configurar o modelo, o treinamento é feito com o seguinte código

```
# Definindo o modelo, a função de perda e o otimizador
model = PlantDiseaseCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop de treinamento
for epoch in range(10):  # Número de épocas
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


```

#### Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.



