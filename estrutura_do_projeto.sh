#!/bin/bash

# Nome do projeto
PROJECT_NAME="smartfarmai"

echo "🌱 Criando estrutura do projeto $PROJECT_NAME..."

# Estrutura de diretórios
mkdir -p $PROJECT_NAME/src/data_processing
mkdir -p $PROJECT_NAME/src/models
mkdir -p $PROJECT_NAME/src/iot
mkdir -p $PROJECT_NAME/src/monitoring
mkdir -p $PROJECT_NAME/config
mkdir -p $PROJECT_NAME/datasets/images
mkdir -p $PROJECT_NAME/datasets/sensor_data
mkdir -p $PROJECT_NAME/notebooks

# Arquivos Python
touch $PROJECT_NAME/src/data_processing/image_preprocessor.py
touch $PROJECT_NAME/src/data_processing/sensor_data_processor.py
touch $PROJECT_NAME/src/models/plant_cnn.py
touch $PROJECT_NAME/src/models/model_trainer.py
touch $PROJECT_NAME/src/iot/sensor_interface.py
touch $PROJECT_NAME/src/monitoring/real_time_monitor.py

# Arquivos de configuração e notebooks
touch $PROJECT_NAME/config/config.yaml
touch $PROJECT_NAME/notebooks/exploratory_analysis.ipynb

echo "✅ Estrutura criada com sucesso!"
tree $PROJECT_NAME
