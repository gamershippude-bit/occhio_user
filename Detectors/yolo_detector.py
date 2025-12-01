"""
yolo_detector.py - Detector YOLO para Cloud Run
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO
import os

logger = logging.getLogger("YOLO-Detector")

class YOLODetector:
    def __init__(self, model_path=None):
        self.model = None
        self.classes = None
        
        try:
            logger.info("🔄 Inicializando YOLO...")
            
            # Se não especificar modelo, usar YOLOv8n (mais leve)
            if model_path is None:
                logger.info("📦 Usando YOLOv8n pré-treinado (nano)")
                self.model = YOLO('yolov8n.pt')
            elif os.path.exists(model_path):
                logger.info(f"📦 Carregando modelo: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.warning(f"⚠️ Modelo não encontrado: {model_path}")
                logger.info("📦 Usando YOLOv8n como fallback")
                self.model = YOLO('yolov8n.pt')
            
            # Obter nomes das classes
            if hasattr(self.model, 'names'):
                self.classes = self.model.names
                logger.info(f"✅ YOLO inicializado com {len(self.classes)} classes")
            else:
                # Classes padrão do COCO
                self.classes = self._get_coco_classes()
                logger.info(f"✅ YOLO inicializado com classes COCO ({len(self.classes)} classes)")
            
            logger.info("🎯 YOLO pronto para detecções")
            
        except Exception as e:
            logger.error(f"❌ ERRO ao inicializar YOLO: {e}")
            raise
    
    def _get_coco_classes(self):
        """Retorna classes COCO padrão"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detectar_com_bbox(self, frame, confidence_threshold=0.25):
        """Detecta objetos retornando bounding boxes"""
        try:
            if self.model is None:
                logger.error("❌ Modelo YOLO não inicializado")
                return []
            
            # Executar inferência
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        class_name = self.classes[cls_id] if cls_id < len(self.classes) else f'class_{cls_id}'
                        
                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': {
                                'x': int(box[0]),
                                'y': int(box[1]),
                                'width': int(box[2] - box[0]),
                                'height': int(box[3] - box[1])
                            }
                        })
            
            logger.info(f"🔍 YOLO detectou {len(detections)} objetos")
            return detections
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção YOLO: {e}")
            return []
    
    def detectar_objetos_rapido(self, frame, confidence_threshold=0.25):
        """Detecta objetos retornando apenas nomes (mais rápido)"""
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            objetos = []
            confiancas = []
            
            for det in detections:
                objetos.append(det['class'])
                confiancas.append(det['confidence'])
            
            return objetos, confiancas
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção rápida: {e}")
            return [], []
    
    def detectar_objetos_filtrados(self, frame, classes_filtro=None, confidence_threshold=0.25):
        """Detecta objetos filtrando por classes específicas"""
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            if classes_filtro:
                # Converter para minúsculas para comparação
                classes_filtro_lower = [c.lower() for c in classes_filtro]
                detections = [
                    d for d in detections 
                    if d['class'].lower() in classes_filtro_lower
                ]
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção filtrada: {e}")
            return []