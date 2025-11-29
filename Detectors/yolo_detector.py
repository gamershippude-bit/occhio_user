"""
YOLOv8 Otimizado - Detecção mais rápida
"""

import cv2
from ultralytics import YOLO
import logging
import numpy as np
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, modelo_path="yolov8n.pt", conf_threshold=0.5, iou_threshold=0.5):  # Confidence menor para mais detecções
        try:
            logger.info(f"Inicializando YOLO otimizado: {modelo_path}")
            
            # Carregar modelo uma vez
            self.modelo = YOLO(modelo_path)
            
            # Configurações otimizadas
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            
            # Cache de classes para evitar acesso repetido
            self.nomes_classes = self.modelo.names
            
            # Sistema de contagem simplificado
            self.contagem_acumulada = defaultdict(int)
            self.frame_count = 0
            
            logger.info("✅ YOLOv8 otimizado carregado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar YOLO: {e}")
            raise

    def detectar_objetos_rapido(self, frame):
        """
        Versão super otimizada para detecção
        """
        try:
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return [], self.contagem_acumulada.copy()

            self.frame_count += 1
            
            # Detecção com configurações otimizadas
            resultados = self.modelo.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                max_det=20,  # Limitar detecções
                agnostic_nms=True  # NMS mais rápido
            )

            objetos_detectados_agora = []

            if resultados and len(resultados) > 0:
                r = resultados[0]
                
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        
                        # Usar cache de nomes
                        nome_classe = self.nomes_classes[cls]

                        if conf >= self.conf_threshold:
                            objetos_detectados_agora.append(nome_classe)

            # Atualizar contagem de forma simples
            self._atualizar_contagem_simples(objetos_detectados_agora)
            
            return objetos_detectados_agora, self.contagem_acumulada.copy()

        except Exception as e:
            logger.error(f"YOLO: Erro na detecao rapida: {e}")
            return [], self.contagem_acumulada.copy()

    def _atualizar_contagem_simples(self, objetos_detectados_agora):
        """Atualização de contagem simplificada"""
        contagem_atual = defaultdict(int)
        for obj in objetos_detectados_agora:
            contagem_atual[obj] += 1
        
        # Suavizar transições (evitar flickering)
        for obj, count in contagem_atual.items():
            current = self.contagem_acumulada.get(obj, 0)
            # Média móvel simples
            if count > current:
                self.contagem_acumulada[obj] = count
            elif current > 0:
                # Diminuir gradualmente
                self.contagem_acumulada[obj] = max(count, current - 1)
        
        # Limpar objetos não detectados há muito tempo
        objetos_ativos = set(objetos_detectados_agora)
        for obj in list(self.contagem_acumulada.keys()):
            if obj not in objetos_ativos and self.contagem_acumulada[obj] <= 1:
                del self.contagem_acumulada[obj]