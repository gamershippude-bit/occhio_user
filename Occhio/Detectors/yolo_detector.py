"""
Módulo de Detecção de Objetos usando YOLO
Este módulo implementa a detecção de objetos utilizando o modelo YOLO,
fornecendo funcionalidades para:
- Inicialização do modelo YOLO
- Detecção de objetos em frames
- Contagem e classificação de objetos
- Visualização dos resultados
"""

import cv2
from ultralytics import YOLO
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self):
        """
        Inicializa o detector YOLO.
        """
        try:
            # Carrega o modelo YOLO com configurações otimizadas
            self.modelo = YOLO('yolov8n.pt')
            self.conf_threshold = 0.6  # Aumentando threshold para menos detecções
            self.iou_threshold = 0.5
            logger.info("Detector YOLO inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar YOLO: {e}")
            raise

    def detectar_objetos(self, frame):
        """
        Detecta objetos no frame usando YOLO.
        
        Args:
            frame: Frame de vídeo para detecção
            
        Returns:
            frame: Frame com detecções desenhadas
            contagem: Dicionário com contagem de objetos
            status: Status da detecção
        """
        try:
            # Processa o frame com YOLO usando configurações otimizadas
            resultados = self.modelo(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            contagem = {}
            status = "Nenhum objeto detectado"
            
            # Processa apenas o primeiro resultado (mais rápido)
            if resultados:
                r = resultados[0]
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    nome_classe = self.modelo.names[cls]
                    
                    if nome_classe not in contagem:
                        contagem[nome_classe] = 0
                    contagem[nome_classe] += 1
                    
                    # Desenha bounding box apenas para objetos com alta confiança
                    if conf > 0.7:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{nome_classe} {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if contagem:
                status = "Objetos detectados"
            
            return frame, contagem, status
            
        except Exception as e:
            logger.error(f"Erro ao detectar objetos: {e}")
            return frame, {}, "Erro na detecção"

    def _desenhar_caixa(self, frame, coords, label, conf):
        """
        Desenha a caixa delimitadora e rótulo no frame.
        
        Args:
            frame: Frame onde desenhar
            coords: Coordenadas da caixa (x1, y1, x2, y2)
            label: Nome da classe detectada
            conf: Nível de confiança da detecção
        """
        x1, y1, x2, y2 = coords
        
        # Desenha retângulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepara texto
        texto = f"{label} {conf:.2f}"
        
        # Calcula tamanho do texto
        (text_width, text_height), _ = cv2.getTextSize(
            texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Desenha fundo para o texto
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1
        )
        
        # Desenha texto
        cv2.putText(
            frame,
            texto,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

# Instância global do detector
detector = YOLODetector()

def detectar_objetos(frame, modelo_path="yolov8n.pt"):
    """
    Função de conveniência para detecção de objetos.
    
    Args:
        frame: Frame para detecção
        modelo_path: Caminho para o modelo YOLO
        
    Returns:
        Resultados da detecção
    """
    global detector
    if detector.modelo is None:
        detector = YOLODetector()
    return detector.detectar_objetos(frame)
