"""
yolo_detector.py - Detector YOLO MELHORADO para Cloud Run
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
        self.class_id_to_name = {}
        
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
                # Criar mapeamento de ID para nome
                self.class_id_to_name = {i: name for i, name in enumerate(self.classes)}
                logger.info(f"✅ YOLO inicializado com {len(self.classes)} classes")
            else:
                # Classes padrão do COCO
                self.classes = self._get_coco_classes()
                self.class_id_to_name = {i: name for i, name in enumerate(self.classes)}
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
        """Detecta objetos retornando bounding boxes - VERSÃO MELHORADA"""
        try:
            if self.model is None:
                logger.error("❌ Modelo YOLO não inicializado")
                return []
            
            # Redimensionar para melhor performance (mantendo proporção)
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                logger.debug(f"📏 Frame redimensionado: {new_w}x{new_h}")
            
            # Executar inferência
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if conf < confidence_threshold:
                            continue
                            
                        class_name = self.classes[cls_id] if cls_id < len(self.classes) else f'class_{cls_id}'
                        
                        # FILTRO ESPECIAL PARA "SPORTS BALL" - aumentar confiança mínima
                        if class_name == 'sports ball':
                            # Sports ball tem muitos falsos positivos - exigir confiança mais alta
                            min_confidence = max(confidence_threshold, 0.5)  # Mínimo 50% para bola
                            if conf < min_confidence:
                                logger.debug(f"⏩ Ignorando sports ball com confiança baixa: {conf:.2f}")
                                continue
                        
                        # FILTRO ESPECIAL PARA OBJETOS PEQUENOS/DIFÍCEIS
                        box_width = box[2] - box[0]
                        box_height = box[3] - box[1]
                        
                        # Ignorar bounding boxes muito pequenas (possíveis falsos positivos)
                        if box_width < 20 or box_height < 20:
                            logger.debug(f"⏩ Ignorando objeto muito pequeno: {class_name} ({box_width}x{box_height})")
                            continue
                        
                        # Adicionar à lista de detecções
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
            
            # Ordenar por confiança (maior primeiro)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Remover duplicatas (mesma classe com overlap)
            detections = self._remover_duplicatas(detections)
            
            logger.info(f"🔍 YOLO detectou {len(detections)} objetos")
            if detections:
                for i, det in enumerate(detections[:5]):  # Mostrar apenas 5 primeiros
                    logger.info(f"   [{i+1}] {det['class']}: {det['confidence']:.3f}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção YOLO: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []
    
    def _remover_duplicatas(self, detections, iou_threshold=0.5):
        """Remove detecções duplicadas com alto overlap"""
        if not detections:
            return detections
        
        filtered = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
            
            filtered.append(detections[i])
            used[i] = True
            
            # Comparar com outras detecções
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                
                # Calcular IOU (Intersection Over Union)
                iou = self._calculate_iou(detections[i]['bbox'], detections[j]['bbox'])
                
                # Se for a mesma classe e IOU alto, é duplicata
                if (detections[i]['class'] == detections[j]['class'] and 
                    iou > iou_threshold):
                    used[j] = True
                    logger.debug(f"⏩ Removendo duplicata: {detections[j]['class']} (IOU: {iou:.2f})")
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calcula Intersection Over Union entre duas bounding boxes"""
        # Coordenadas da interseção
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
        
        # Área da interseção
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Áreas individuais
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        
        # IOU
        iou = intersection_area / (area1 + area2 - intersection_area)
        return iou
    
    def detectar_objetos_filtrados(self, frame, classes_filtro=None, confidence_threshold=0.25):
        """Detecta objetos filtrando por classes específicas"""
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            if classes_filtro:
                # Converter para minúsculas para comparação
                classes_filtro_lower = [c.lower() for c in classes_filtro]
                filtered_detections = []
                
                for d in detections:
                    if d['class'].lower() in classes_filtro_lower:
                        filtered_detections.append(d)
                    else:
                        # Verificar se é um sinônimo
                        synonyms = self._get_synonyms(d['class'])
                        if any(syn.lower() in classes_filtro_lower for syn in synonyms):
                            filtered_detections.append(d)
                
                detections = filtered_detections
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção filtrada: {e}")
            return []
    
    def _get_synonyms(self, class_name):
        """Retorna sinônimos para uma classe"""
        synonyms = {
            'person': ['people', 'human'],
            'car': ['vehicle', 'automobile'],
            'tv': ['television', 'monitor'],
            'cell phone': ['mobile phone', 'phone', 'smartphone'],
            'laptop': ['notebook', 'computer'],
            'sports ball': ['ball', 'bola'],
            'chair': ['seat'],
            'couch': ['sofa', 'settee'],
            'potted plant': ['plant', 'flower'],
            'dining table': ['table', 'desk'],
            'book': ['books']
        }
        
        return synonyms.get(class_name, [class_name])
    
    def detectar_objetos_agrupados(self, frame, confidence_threshold=0.25):
        """Detecta objetos e agrupa por tipo com contagem"""
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            # Agrupar por classe
            grouped = {}
            for det in detections:
                class_name = det['class']
                if class_name not in grouped:
                    grouped[class_name] = []
                grouped[class_name].append(det)
            
            # Criar lista de objetos agrupados
            objetos_agrupados = []
            for class_name, det_list in grouped.items():
                objetos_agrupados.append({
                    'name': class_name,
                    'count': len(det_list),
                    'confidence_avg': sum(d['confidence'] for d in det_list) / len(det_list),
                    'detections': det_list
                })
            
            return objetos_agrupados
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção agrupada: {e}")
            return []
    
    def detectar_objetos_yolo(self, frame, confidence_threshold=0.25):
        """Método de compatibilidade com versão antiga"""
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            # Extrair apenas nomes das classes
            objetos = [d['class'] for d in detections]
            confiancas = [d['confidence'] for d in detections]
            
            return objetos, confiancas
            
        except Exception as e:
            logger.error(f"❌ Erro detectar_objetos_yolo: {e}")
            return [], []
    
    def testar_deteccao(self, frame):
        """Método de teste simples"""
        try:
            logger.info("🧪 Testando detecção YOLO...")
            
            # Redimensionar para melhor performance
            if frame.shape[0] > 640:
                scale = 640 / frame.shape[0]
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Detectar
            detections = self.detectar_com_bbox(frame, confidence_threshold=0.15)
            
            # Log detalhado
            logger.info(f"📊 Resultado do teste:")
            for i, det in enumerate(detections[:5]):  # Mostrar apenas 5 primeiros
                logger.info(f"  {i+1}. {det['class']}: {det['confidence']:.2f} at ({det['bbox']['x']}, {det['bbox']['y']})")
            
            return len(detections)
            
        except Exception as e:
            logger.error(f"❌ Erro no teste YOLO: {e}")
            return 0