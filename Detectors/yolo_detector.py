"""
yolo_detector.py
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO
import os
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("YOLO-Detector")

class YOLODetector:
    def __init__(self, model_path: Optional[str] = None, use_custom: bool = True, device: str = 'cpu'):
        """
        Inicializa o detector YOLO.
        
        Args:
            model_path: Caminho para modelo customizado (opcional)
            use_custom: Se deve usar modelo customizado se disponível
            device: Dispositivo para inferência ('cpu' ou 'cuda')
        """
        self.model = None
        self.classes = None
        self.class_id_to_name = {}
        self.device = device
        self._ultimo_estado_log: set = set()
        
        try:
            logger.info("🔄 Inicializando YOLO...")
            
            # Configurar modelo baseado nas opções
            if model_path is None or not os.path.exists(model_path):
                logger.info("📦 Usando YOLOv8s pré-treinado (small - melhor precisão)")
                self.model = YOLO('yolov8s.pt').to(self.device)
            elif os.path.exists(model_path) and use_custom:
                logger.info(f"📦 Carregando modelo customizado: {model_path}")
                self.model = YOLO(model_path).to(self.device)
            else:
                logger.warning("⚠️ Usando YOLOv8s como fallback")
                self.model = YOLO('yolov8s.pt').to(self.device)
            
            # Configurar classes
            self._setup_classes()
            
            logger.info(f"🎯 YOLO inicializado com sucesso no dispositivo: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ ERRO ao inicializar YOLO: {e}")
            raise
    
    def _setup_classes(self) -> None:
        """Configura as classes do modelo"""
        try:
            if hasattr(self.model, 'names') and self.model.names:
                self.classes = self.model.names
                self.class_id_to_name = {i: name for i, name in enumerate(self.classes)}
                logger.info(f"✅ YOLO com {len(self.classes)} classes customizadas")
            else:
                # Classes padrão do COCO
                self.classes = self._get_coco_classes()
                self.class_id_to_name = {i: name for i, name in enumerate(self.classes)}
                logger.info(f"✅ YOLO com classes COCO ({len(self.classes)} classes)")
            
            # Log das primeiras classes disponíveis
            if isinstance(self.classes, dict):
                class_list = list(self.classes.values())[:10]
            else:
                class_list = self.classes[:10]
            logger.debug(f"📋 Primeiras 10 classes: {class_list}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao configurar classes: {e}")
            self.classes = self._get_coco_classes()
            self.class_id_to_name = {i: name for i, name in enumerate(self.classes)}
    
    def _get_coco_classes(self) -> List[str]:
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
    
    def preprocess_frame(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """
        Pré-processa o frame para melhor detecção.
        
        Args:
            frame: Frame de entrada (BGR)
            target_size: Tamanho máximo para redimensionamento
            
        Returns:
            Frame pré-processado
        """
        try:
            if frame is None or frame.size == 0:
                logger.error("❌ Frame vazio ou None")
                return frame
            
            h, w = frame.shape[:2]
            processed_frame = frame.copy()
            
            # 1. Melhorar contraste com CLAHE (apenas para imagens coloridas)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                try:
                    # Converter para LAB
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l_channel, a, b = cv2.split(lab)
                    
                    # Aplicar CLAHE no canal L
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    l_channel = clahe.apply(l_channel)
                    
                    # Mesclar canais e converter de volta
                    lab = cv2.merge([l_channel, a, b])
                    processed_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    
                    logger.debug("✨ Contraste ajustado com CLAHE")
                except Exception as e:
                    logger.warning(f"⚠️ Erro no CLAHE: {e}")
            
            # 2. Redimensionar mantendo proporção
            if max(h, w) > target_size:
                scale = target_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                processed_frame = cv2.resize(processed_frame, (new_w, new_h), 
                                           interpolation=cv2.INTER_AREA)
                logger.debug(f"📏 Frame redimensionado: {new_w}x{new_h}")
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento: {e}")
            return frame
    
    def detectar_com_bbox(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detecta objetos retornando bounding boxes.
        
        Args:
            frame: Frame de entrada (BGR)
            confidence_threshold: Threshold de confiança mínimo
            
        Returns:
            Lista de detecções com bounding boxes
        """
        try:
            if self.model is None:
                logger.error("❌ Modelo YOLO não inicializado")
                return []
            
            if frame is None or frame.size == 0:
                logger.error("❌ Frame de entrada vazio")
                return []
            
            original_h, original_w = frame.shape[:2]
            
            # Pré-processamento
            processed_frame = self.preprocess_frame(frame.copy())
            processed_h, processed_w = processed_frame.shape[:2]
            
            # Fatores de escala para coordenadas originais
            scale_x = original_w / processed_w
            scale_y = original_h / processed_h
            
            # Executar inferência com parâmetros otimizados
            results = self.model(
                processed_frame,
                conf=confidence_threshold,
                iou=0.45,           # Reduzir sobreposição
                max_det=100,        # Limitar detecções
                verbose=False,
                device=self.device
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Obter nome da classe
                        class_name = self._get_class_name(cls_id)
                        
                        # Aplicar filtros
                        if not self._passes_filters(class_name, conf, box, scale_x, scale_y, 
                                                   original_w, original_h, detections):
                            continue
                        
                        # Converter coordenadas para tamanho original
                        x1 = int(box[0] * scale_x)
                        y1 = int(box[1] * scale_y)
                        x2 = int(box[2] * scale_x)
                        y2 = int(box[3] * scale_y)
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        
                        # Adicionar detecção
                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': {
                                'x': x1,
                                'y': y1,
                                'width': box_width,
                                'height': box_height,
                                'area': box_area,
                                'center_x': x1 + box_width // 2,
                                'center_y': y1 + box_height // 2
                            }
                        })
            
            # Pós-processamento
            detections = self._post_process_detections(detections)
            
            # Log com debounce — só quando o conjunto de classes muda
            estado_atual = set(d['class'] for d in detections)
            if estado_atual != self._ultimo_estado_log:
                if estado_atual:
                    logger.info(
                        f"🔍 YOLO detectou {len(detections)} objeto(s): {', '.join(sorted(estado_atual))}"
                    )
                else:
                    logger.info('🔍 Cena limpa — nenhum objeto detectado')
                self._ultimo_estado_log = estado_atual
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção YOLO: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []
    
    def _get_class_name(self, cls_id: int) -> str:
        """Obtém nome da classe de forma segura"""
        try:
            if isinstance(self.classes, dict):
                return self.classes.get(int(cls_id), f'class_{cls_id}')
            elif isinstance(self.classes, list):
                if 0 <= int(cls_id) < len(self.classes):
                    return self.classes[int(cls_id)]
                else:
                    return f'class_{cls_id}'
            else:
                return f'class_{cls_id}'
        except:
            return f'class_{cls_id}'
    
    def _passes_filters(self, class_name: str, confidence: float, box: np.ndarray,
                       scale_x: float, scale_y: float, img_w: int, img_h: int,
                       existing_detections: List[Dict]) -> bool:
        """
        Aplica todos os filtros para decidir se mantém a detecção.
        
        Returns:
            True se a detecção passar todos os filtros
        """
        try:
            # Converter coordenadas para tamanho original
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # 1. FILTRO: Confiança mínima por classe
            class_min_confidences = {
                # Classes com muitos falsos positivos (aumentar muito)
                'sports ball': 0.75,
                'tie': 0.70,
                'handbag': 0.65,
                'backpack': 0.65,
                'remote': 0.65,
                'kite': 0.75,
                'frisbee': 0.70,
                'skateboard': 0.65,
                'surfboard': 0.65,
                'airplane': 0.70,
                'toothbrush': 0.65,
                'hair drier': 0.65,
                'traffic light': 0.60,
                'stop sign': 0.60,
                
                # Classes comuns (threshold moderado)
                'person': 0.40,
                'car': 0.45,
                'chair': 0.45,
                'bottle': 0.55,
                'cup': 0.55,
                'book': 0.45,
                'laptop': 0.50,
                'cell phone': 0.60,
                'tv': 0.55,
                'keyboard': 0.50,
                'mouse': 0.50,
                'dining table': 0.45,
                'couch': 0.45,
                'potted plant': 0.50,
            }
            
            min_conf = class_min_confidences.get(class_name, 0.5)
            if confidence < min_conf:
                logger.debug(f"⏩ Filtro confiança: {class_name} ({confidence:.3f} < {min_conf})")
                return False
            
            # 2. FILTRO: Tamanho mínimo (em pixels e % da imagem)
            img_area = img_w * img_h
            min_area_ratio = 0.0003  # 0.03% da área da imagem
            
            if area < (img_area * min_area_ratio):
                logger.debug(f"⏩ Filtro tamanho: {class_name} muito pequeno ({area} pixels)")
                return False
            
            # Tamanhos mínimos absolutos por classe
            class_min_sizes = {
                'person': (30, 80),
                'car': (40, 30),
                'cell phone': (15, 25),
                'bottle': (10, 20),
                'chair': (25, 40),
                'laptop': (30, 20),
                'tv': (40, 30),
                'book': (15, 20),
            }
            
            min_w, min_h = class_min_sizes.get(class_name, (20, 20))
            if width < min_w or height < min_h:
                logger.debug(f"⏩ Filtro tamanho mínimo: {class_name} {width}x{height} < {min_w}x{min_h}")
                return False
            
            # 3. FILTRO: Razão aspecto (proporção largura/altura)
            if height > 0:
                aspect_ratio = width / height
                
                # Razões aspecto esperadas por classe
                expected_aspects = {
                    'person': (0.2, 0.6),      # Pessoas são mais altas
                    'car': (0.8, 3.0),         # Carros são mais largos
                    'cell phone': (0.4, 0.8),  # Celulares verticais
                    'bottle': (0.2, 0.6),      # Garrafas são mais altas
                    'laptop': (1.2, 2.5),      # Laptops são mais largos
                    'tv': (1.0, 2.0),         # TVs são retangulares
                    'book': (0.5, 1.5),       # Livros variam
                }
                
                if class_name in expected_aspects:
                    min_ar, max_ar = expected_aspects[class_name]
                    if aspect_ratio < min_ar or aspect_ratio > max_ar:
                        logger.debug(f"⏩ Filtro aspecto: {class_name} razão {aspect_ratio:.2f} fora do esperado")
                        return False
            
            # 4. FILTRO: Contexto (objetos improváveis no cenário)
            if self._is_improbable_context(class_name, existing_detections):
                logger.debug(f"⏩ Filtro contexto: {class_name} improvável")
                return False
            
            # 5. FILTRO: Posição na imagem (alguns objetos não devem estar nas bordas)
            if self._is_invalid_position(class_name, x1, y1, x2, y2, img_w, img_h):
                logger.debug(f"⏩ Filtro posição: {class_name} em posição inválida")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro nos filtros: {e}")
            return False
    
    def _is_improbable_context(self, class_name: str, existing_detections: List[Dict]) -> bool:
        """Verifica se a classe faz sentido no contexto atual"""
        # Objetos tipicamente de exterior
        outdoor_objects = ['airplane', 'boat', 'surfboard', 'kite', 'traffic light', 
                          'stop sign', 'fire hydrant', 'parking meter']
        
        # Objetos tipicamente de interior
        indoor_objects = ['tv', 'laptop', 'remote', 'keyboard', 'mouse', 'microwave',
                         'oven', 'toaster', 'refrigerator', 'toilet', 'bed']
        
        # Se for objeto de exterior, verificar contexto
        if class_name in outdoor_objects:
            indoor_count = sum(1 for d in existing_detections if d['class'] in indoor_objects)
            # Se já detectamos objetos de interior, objeto de exterior é improvável
            if indoor_count > 2:
                return True
        
        return False
    
    def _is_invalid_position(self, class_name: str, x1: int, y1: int, x2: int, y2: int, 
                            img_w: int, img_h: int) -> bool:
        """Verifica se a posição na imagem é válida para a classe"""
        # Calcular distância até as bordas
        from_left = x1
        from_right = img_w - x2
        from_top = y1
        from_bottom = img_h - y2
        
        border_threshold = 10  # pixels
        
        # Alguns objetos não devem estar colados nas bordas
        if class_name in ['person', 'car', 'chair', 'dining table', 'couch']:
            if (from_left < border_threshold or from_right < border_threshold or
                from_top < border_threshold or from_bottom < border_threshold):
                return True
        
        return False
    
    def _post_process_detections(self, detections: List[Dict]) -> List[Dict]:
        """Aplica pós-processamento às detecções"""
        if not detections:
            return detections
        
        # 1. Ordenar por confiança (decrescente)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 2. Remover duplicatas por NMS (Non-Maximum Suppression)
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        # 3. Limitar número máximo de detecções
        max_detections = 20
        if len(detections) > max_detections:
            logger.debug(f"📉 Limitando de {len(detections)} para {max_detections} detecções")
            detections = detections[:max_detections]
        
        return detections
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Aplica Non-Maximum Suppression para remover caixas sobrepostas"""
        if len(detections) <= 1:
            return detections
        
        # Separar por classe
        class_groups = {}
        for det in detections:
            class_name = det['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        # Aplicar NMS para cada classe
        filtered_detections = []
        
        for class_name, class_dets in class_groups.items():
            # Ordenar por confiança
            class_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            while class_dets:
                # Manter a detecção mais confiante
                best_det = class_dets.pop(0)
                filtered_detections.append(best_det)
                
                # Remover detecções com alto IOU
                to_remove = []
                for i, det in enumerate(class_dets):
                    iou = self._calculate_iou(best_det['bbox'], det['bbox'])
                    if iou > iou_threshold:
                        to_remove.append(i)
                
                # Remover em ordem reversa
                for i in reversed(to_remove):
                    removed = class_dets.pop(i)
                    logger.debug(f"⏩ NMS removido: {removed['class']} (IOU: {iou:.2f})")
        
        return filtered_detections
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calcula Intersection Over Union entre duas bounding boxes"""
        try:
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
            
            # Evitar divisão por zero
            union_area = area1 + area2 - intersection_area
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
            
        except:
            return 0.0
    
    def detectar_objetos_filtrados(self, frame: np.ndarray, 
                                  classes_filtro: Optional[List[str]] = None,
                                  confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detecta objetos filtrando por classes específicas.
        
        Args:
            frame: Frame de entrada
            classes_filtro: Lista de classes para filtrar
            confidence_threshold: Threshold de confiança
            
        Returns:
            Lista de detecções filtradas
        """
        try:
            # Primeiro detectar todos os objetos
            all_detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            if not classes_filtro or not all_detections:
                return all_detections
            
            # Converter classes_filtro para minúsculas
            classes_filtro_lower = [c.lower() for c in classes_filtro]
            
            # Filtrar detecções
            filtered = []
            for det in all_detections:
                class_name_lower = det['class'].lower()
                
                # Verificar se a classe está na lista
                if class_name_lower in classes_filtro_lower:
                    filtered.append(det)
                else:
                    # Verificar sinônimos
                    synonyms = self._get_synonyms(det['class'])
                    if any(syn.lower() in classes_filtro_lower for syn in synonyms):
                        filtered.append(det)
            
            logger.info(f"🎯 Filtradas {len(filtered)} detecções de {len(all_detections)} total")
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção filtrada: {e}")
            return []
    
    def _get_synonyms(self, class_name: str) -> List[str]:
        """Retorna sinônimos para uma classe"""
        synonyms = {
            'person': ['people', 'human', 'man', 'woman', 'child'],
            'car': ['vehicle', 'automobile', 'auto', 'sedan', 'truck'],
            'tv': ['television', 'monitor', 'screen'],
            'cell phone': ['mobile phone', 'phone', 'smartphone', 'mobile'],
            'laptop': ['notebook', 'computer', 'macbook'],
            'sports ball': ['ball', 'bola', 'football', 'soccer ball'],
            'chair': ['seat', 'stool'],
            'couch': ['sofa', 'settee', 'divan'],
            'potted plant': ['plant', 'flower', 'vase', 'indoor plant'],
            'dining table': ['table', 'desk', 'worktable'],
            'book': ['books', 'notebook', 'magazine'],
            'bottle': ['water bottle', 'glass bottle', 'plastic bottle'],
            'cup': ['mug', 'glass', 'tumbler'],
        }
        
        return synonyms.get(class_name, [class_name])
    
    def detectar_objetos_agrupados(self, frame: np.ndarray, 
                                  confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detecta objetos e agrupa por tipo com contagem.
        
        Args:
            frame: Frame de entrada
            confidence_threshold: Threshold de confiança
            
        Returns:
            Lista de objetos agrupados
        """
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            # Agrupar por classe
            grouped_dict = {}
            for det in detections:
                class_name = det['class']
                if class_name not in grouped_dict:
                    grouped_dict[class_name] = {
                        'name': class_name,
                        'count': 0,
                        'total_confidence': 0.0,
                        'detections': []
                    }
                
                grouped_dict[class_name]['count'] += 1
                grouped_dict[class_name]['total_confidence'] += det['confidence']
                grouped_dict[class_name]['detections'].append(det)
            
            # Converter para lista e calcular média
            grouped_list = []
            for class_name, data in grouped_dict.items():
                data['confidence_avg'] = data['total_confidence'] / data['count']
                del data['total_confidence']  # Remover campo temporário
                grouped_list.append(data)
            
            # Ordenar por contagem (decrescente)
            grouped_list.sort(key=lambda x: x['count'], reverse=True)
            
            logger.info(f"📊 Agrupadas {len(grouped_list)} classes de {len(detections)} detecções")
            return grouped_list
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção agrupada: {e}")
            return []
    
    def detectar_objetos_yolo(self, frame: np.ndarray, 
                             confidence_threshold: float = 0.5) -> Tuple[List[str], List[float]]:
        """
        Método de compatibilidade com versão antiga.
        
        Args:
            frame: Frame de entrada
            confidence_threshold: Threshold de confiança
            
        Returns:
            Tuple: (lista de classes, lista de confianças)
        """
        try:
            detections = self.detectar_com_bbox(frame, confidence_threshold)
            
            # Extrair apenas nomes e confianças
            objetos = [d['class'] for d in detections]
            confiancas = [d['confidence'] for d in detections]
            
            return objetos, confiancas
            
        except Exception as e:
            logger.error(f"❌ Erro detectar_objetos_yolo: {e}")
            return [], []
    
    def testar_deteccao(self, frame: np.ndarray, confidence_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Método de teste completo.
        
        Args:
            frame: Frame de entrada
            confidence_thresholds: Lista de thresholds para testar
            
        Returns:
            Dicionário com resultados dos testes
        """
        try:
            logger.info("🧪 Iniciando teste de detecção YOLO...")
            
            if confidence_thresholds is None:
                confidence_thresholds = [0.3, 0.4, 0.5, 0.6]
            
            resultados = {
                'thresholds': {},
                'melhor_threshold': None,
                'total_detections': 0
            }
            
            # Testar diferentes thresholds
            best_score = -1
            best_threshold = None
            
            for conf in confidence_thresholds:
                logger.info(f"🧪 Testando com threshold={conf}")
                detections = self.detectar_com_bbox(frame, confidence_threshold=conf)
                
                # Calcular "score" (número de detecções * confiança média)
                if detections:
                    avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                    score = len(detections) * avg_confidence
                else:
                    score = 0
                
                # Armazenar resultados
                resultados['thresholds'][conf] = {
                    'num_detections': len(detections),
                    'detections': detections[:5],  # Apenas 5 primeiras
                    'score': score
                }
                
                # Atualizar melhor threshold
                if score > best_score:
                    best_score = score
                    best_threshold = conf
            
            resultados['melhor_threshold'] = best_threshold
            
            # Usar melhor threshold para detecção final
            if best_threshold is not None:
                final_detections = self.detectar_com_bbox(frame, confidence_threshold=best_threshold)
                resultados['total_detections'] = len(final_detections)
                
                # Log detalhado
                logger.info(f"📊 Resultado final do teste (threshold={best_threshold}):")
                for i, det in enumerate(final_detections[:10]):  # Mostrar até 10
                    bbox = det['bbox']
                    logger.info(f"  {i+1}. {det['class']}: {det['confidence']:.3f} "
                              f"pos:({bbox['x']},{bbox['y']}) size:{bbox['width']}x{bbox['height']}")
            
            logger.info(f"✅ Teste completo. Melhor threshold: {best_threshold}")
            return resultados
            
        except Exception as e:
            logger.error(f"❌ Erro no teste YOLO: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado"""
        info = {
            'model_loaded': self.model is not None,
            'device': self.device,
            'num_classes': len(self.classes) if self.classes else 0,
            'classes_sample': []
        }
        
        if self.classes:
            if isinstance(self.classes, dict):
                info['classes_sample'] = list(self.classes.values())[:10]
            else:
                info['classes_sample'] = self.classes[:10]
        
        return info


# Função auxiliar para criar detector
def criar_detector_yolo(modelo_personalizado: Optional[str] = None, 
                       usar_custom: bool = True,
                       nivel_log: int = logging.INFO) -> YOLODetector:
    """
    Função de fábrica para criar detector YOLO.
    
    Args:
        modelo_personalizado: Caminho para modelo personalizado
        usar_custom: Se deve usar modelo customizado
        nivel_log: Nível de logging
        
    Returns:
        Instância do detector YOLO
    """
    # Configurar logging
    logging.basicConfig(
        level=nivel_log,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        detector = YOLODetector(modelo_personalizado, usar_custom)
        logger.info("✅ Detector YOLO criado com sucesso")
        return detector
    except Exception as e:
        logger.error(f"❌ Falha ao criar detector: {e}")
        raise


# Exemplo de uso
if __name__ == "__main__":
    # Criar detector
    detector = criar_detector_yolo()
    
    # Carregar imagem de teste
    import cv2
    teste_img = cv2.imread("teste.jpg")
    
    if teste_img is not None:
        # Testar detecção
        resultados = detector.testar_deteccao(teste_img)
        
        # Detecção normal
        deteccoes = detector.detectar_com_bbox(teste_img)
        
        print(f"\n📋 RESUMO:")
        print(f"Total de detecções: {len(deteccoes)}")
        for det in deteccoes[:5]:
            print(f"  - {det['class']}: {det['confidence']:.3f}")
    else:
        print("❌ Imagem de teste não encontrada")