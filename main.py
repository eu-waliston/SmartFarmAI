import yaml
import torch
from torchvision import datasets, transforms
from src.data_processing.image_preprocessor import AdvancedImagePreprocessor
from src.models.plant_cnn import AdvancedPlantDiseaseCNN
from src.models.model_trainer import PlantModelTrainer
from src.iot.sensor_interface import SensorNetwork
from src.monitoring.real_time_monitor import RealTimePlantMonitor

def main():
    # Carregar configuração
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Preparar dados
    preprocessor = AdvancedImagePreprocessor(config)

    # Dataset (exemplo - ajuste para seu dataset real)
    train_dataset = datasets.ImageFolder(
        'datasets/images/train',
        transform=lambda x: preprocessor.apply_train_transform(x)
    )

    val_dataset = datasets.ImageFolder(
        'datasets/images/val',
        transform=lambda x: preprocessor.apply_val_transform(x)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False
    )

    # Modelo
    model = AdvancedPlantDiseaseCNN(
        num_classes=len(train_dataset.classes)
    )

    # Treinamento
    trainer = PlantModelTrainer(model, config)
    trainer.train(train_loader, val_loader, config['model']['epochs'])
    trainer.plot_training_history()

    # Salvar modelo
    torch.save(model.state_dict(), 'models/plant_disease_model.pth')

    # Sistema de monitoramento em tempo real
    sensor_network = SensorNetwork()
    class_names = train_dataset.classes

    monitor = RealTimePlantMonitor(
        'models/plant_disease_model.pth',
        sensor_network,
        preprocessor,
        class_names
    )

    print("Sistema de monitoramento agrícola inicializado com sucesso!")

    # Exemplo de uso em tempo real
    # (em produção, isso seria integrado com câmeras/drones)
    try:
        while True:
            # Simular captura de imagem
            # image = capture_image_from_camera()
            # analysis = monitor.comprehensive_analysis(image)
            # print(json.dumps(analysis, indent=2))
            pass

    except KeyboardInterrupt:
        print("Sistema encerrado.")

if __name__ == "__main__":
    main()