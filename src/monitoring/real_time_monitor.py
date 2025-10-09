import torch
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime

class RealTimePlantMonitor:
    def __init__(self, model_path, sensor_network, preprocessor, class_names):
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.sensor_network = sensor_network
        self.preprocessor = preprocessor
        self.class_names = class_names

    def analyze_image(self, image):
        # Pré-processar imagem
        processed_image = self.preprocessor.apply_val_transform(image)
        processed_image = processed_image.unsqueeze(0)

        # Fazer predição
        with torch.no_grad():
            output = self.model(processed_image)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        return {
            'prediction': self.class_names[prediction.item()],
            'confidence': confidence.item(),
            'timestamp': datetime.now().isoformat()
        }

    def analyze_sensor_data(self):
        sensor_data = self.sensor_network.collect_all_data()

        # Análise simples das condições ambientais
        risk_factors = []

        temp = sensor_data['temperature']['value']
        humidity = sensor_data['humidity']['value']
        ph = sensor_data['soil_ph']['value']

        if temp > 30 or temp < 10:
            risk_factors.append(f"Temperatura crítica: {temp}°C")
        if humidity > 80:
            risk_factors.append(f"Umidade alta: {humidity}%")
        if ph < 5.5 or ph > 7.5:
            risk_factors.append(f"pH do solo inadequado: {ph}")

        return {
            'sensor_data': sensor_data,
            'risk_factors': risk_factors,
            'overall_risk': 'ALTA' if risk_factors else 'BAIXA'
        }

    def comprehensive_analysis(self, image):
        image_analysis = self.analyze_image(image)
        sensor_analysis = self.analyze_sensor_data()

        return {
            'image_analysis': image_analysis,
            'sensor_analysis': sensor_analysis,
            'recommendations': self._generate_recommendations(
                image_analysis, sensor_analysis
            )
        }

    def _generate_recommendations(self, image_analysis, sensor_analysis):
        recommendations = []

        # Recomendações baseadas na imagem
        if image_analysis['confidence'] > 0.7:
            if 'doença' in image_analysis['prediction'].lower():
                recommendations.append("Aplicar tratamento específico para a doença identificada")
            elif 'praga' in image_analysis['prediction'].lower():
                recommendations.append("Considerar controle biológico ou pesticidas apropriados")

        # Recomendações baseadas em sensores
        for risk in sensor_analysis['risk_factors']:
            if 'Temperatura' in risk:
                recommendations.append("Ajustar sistema de irrigação ou sombreamento")
            elif 'Umidade' in risk:
                recommendations.append("Melhorar ventilação ou drenagem")
            elif 'pH' in risk:
                recommendations.append("Aplicar corretores de pH no solo")

        if not recommendations:
            recommendations.append("Condições normais - Manter monitoramento regular")

        return recommendations