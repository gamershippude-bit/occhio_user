"""
Módulo de Detecção de Objetos usando YOLOv8 com detecção em TODOS os frames
"""

import cv2
from ultralytics import YOLO
import logging
import numpy as np
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(self, modelo_path="yolov8n.pt", conf_threshold=0.6, iou_threshold=0.5):
        """Inicializa o detector YOLO com detecção em todos os frames."""
        try:
            logger.info(f"Inicializando YOLO com modelo: {modelo_path}")
            self.modelo = YOLO(modelo_path)
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            
            # Sistema de contagem mais estável - mantém por 5 frames
            self.contagem_acumulada = defaultdict(int)
            self.historico_deteccoes = defaultdict(lambda: deque(maxlen=10))
            self.ultima_deteccao = defaultdict(int)
            self.frame_count = 0
            self.ultimo_log = 0
            
            logger.info("YOLOv8 carregado com sistema de contagem estável (5 frames).")
        except Exception as e:
            logger.error(f"Erro ao inicializar YOLO: {e}")
            raise

    def detectar_objetos(self, frame):
        """
        Detecta objetos em TODOS os frames e desenha caixas sempre.
        Retorna: (frame_processado, contagem_frame, contagem_acumulada)
        """
        try:
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.error("YOLO: Frame invalido para detecao")
                return frame, {}, self.contagem_acumulada.copy()

            self.frame_count += 1
            frame_processado = frame.copy()
            
            # ✅ SEMPRE detectar objetos (não pular frames)
            resultados = self.modelo.predict(
                source=frame_processado,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            contagem_frame = {}
            objetos_detectados_agora = []

            if resultados and len(resultados) > 0:
                r = resultados[0]
                
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        nome_classe = self.modelo.names[cls]

                        # Apenas objetos com confiança alta
                        if conf >= self.conf_threshold:
                            contagem_frame[nome_classe] = contagem_frame.get(nome_classe, 0) + 1
                            objetos_detectados_agora.append(nome_classe)
                            
                            # Registrar última detecção
                            self.ultima_deteccao[nome_classe] = self.frame_count
                            
                            # ✅ SEMPRE desenhar caixas nos frames detectados
                            self._desenhar_caixa(
                                frame_processado,
                                tuple(map(int, box.xyxy[0])),
                                nome_classe,
                                conf
                            )

            # ATUALIZAR CONTAGEM COM ESTABILIDADE (mantém por 5 frames)
            self._atualizar_contagem_estavel(objetos_detectados_agora)
            
            # Log a cada 60 frames
            if self.frame_count - self.ultimo_log >= 60:
                self._log_estatisticas()
                self.ultimo_log = self.frame_count

            return frame_processado, contagem_frame, self.contagem_acumulada.copy()

        except Exception as e:
            logger.error(f"YOLO: Erro na detecao: {e}")
            return frame, {}, self.contagem_acumulada.copy()

    def _atualizar_contagem_estavel(self, objetos_detectados_agora):
        """Atualiza a contagem mantendo objetos por 5 frames"""
        
        # 1. Registrar detecções atuais no histórico
        for obj in objetos_detectados_agora:
            self.historico_deteccoes[obj].append(self.frame_count)
        
        # 2. Manter objetos que foram detectados recentemente (últimos 5 frames)
        objetos_ativos = set()
        for obj, deteccoes in self.historico_deteccoes.items():
            if deteccoes:
                # Considerar ativo se foi detectado nos últimos 5 frames
                frames_desde_ultima_deteccao = self.frame_count - max(deteccoes)
                if frames_desde_ultima_deteccao <= 5:
                    objetos_ativos.add(obj)
        
        # 3. Atualizar contagem acumulada apenas para objetos ativos
        nova_contagem = defaultdict(int)
        
        # Primeiro, manter objetos que ainda estão ativos
        for obj in objetos_ativos:
            if obj in objetos_detectados_agora:
                # Se detectado agora, usar contagem atual
                contagem_atual = objetos_detectados_agora.count(obj)
                nova_contagem[obj] = max(self.contagem_acumulada.get(obj, 0), contagem_atual)
            else:
                # Se não detectado agora mas ainda ativo, manter contagem anterior
                nova_contagem[obj] = self.contagem_acumulada.get(obj, 0)
        
        # 4. Adicionar novos objetos detectados agora
        for obj in set(objetos_detectados_agora):
            if obj not in nova_contagem:
                contagem_atual = objetos_detectados_agora.count(obj)
                nova_contagem[obj] = contagem_atual
        
        # 5. Limpar objetos inativos (não detectados há mais de 5 frames)
        objetos_para_limpar = []
        for obj in self.contagem_acumulada:
            if obj not in nova_contagem:
                objetos_para_limpar.append(obj)
        
        for obj in objetos_para_limpar:
            if obj in self.historico_deteccoes:
                del self.historico_deteccoes[obj]
            if obj in self.ultima_deteccao:
                del self.ultima_deteccao[obj]
        
        # 6. Atualizar contagem acumulada
        self.contagem_acumulada = nova_contagem

    def _log_estatisticas(self):
        """Log das estatísticas atuais"""
        if self.contagem_acumulada:
            total_objetos = sum(self.contagem_acumulada.values())
            pessoas = self.contagem_acumulada.get('person', 0)
            
            logger.info(f"📊 YOLO - Pessoas: {pessoas} | Total objetos: {total_objetos}")
            
            if total_objetos > 0:
                detalhes = ", ".join([f"{obj}({count})" for obj, count in self.contagem_acumulada.items()])
                logger.info(f"📋 YOLO Detalhes: {detalhes}")

    def _desenhar_caixa(self, frame, coords, label, conf):
        """Desenha uma caixa e o rotulo sobre o objeto."""
        try:
            x1, y1, x2, y2 = coords
            texto = f"{label} {conf:.2f}"

            # Cores diferentes por categoria
            if label == 'person':
                cor = (0, 255, 0)  # Verde para pessoas
            elif label in ['chair', 'couch', 'bed', 'table', 'dining table']:
                cor = (255, 165, 0)  # Laranja para móveis
            elif label in ['laptop', 'tv', 'cell phone', 'monitor']:
                cor = (0, 255, 255)  # Amarelo para eletrônicos
            else:
                cor = (255, 0, 0)  # Vermelho para outros

            # Caixa mais espessa para melhor visibilidade
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 3)
            
            # Fundo do texto
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), cor, -1)
            
            # Texto mais legível
            cv2.putText(frame, texto, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
        except Exception as e:
            logger.error(f"YOLO: Erro ao desenhar caixa: {e}")

    def detectar_objetos_rapido(self, frame):
        """
        Versão otimizada para detecção em tempo real.
        Retorna apenas dados, sem processar frame.
        """
        try:
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return [], self.contagem_acumulada.copy()

            self.frame_count += 1
            
            # Detecção rápida
            resultados = self.modelo.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            objetos_detectados_agora = []

            if resultados and len(resultados) > 0:
                r = resultados[0]
                
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        nome_classe = self.modelo.names[cls]

                        if conf >= self.conf_threshold:
                            objetos_detectados_agora.append(nome_classe)
                            self.ultima_deteccao[nome_classe] = self.frame_count

            # Atualizar contagem
            self._atualizar_contagem_estavel(objetos_detectados_agora)
            
            return objetos_detectados_agora, self.contagem_acumulada.copy()

        except Exception as e:
            logger.error(f"YOLO: Erro na detecao rapida: {e}")
            return [], self.contagem_acumulada.copy()

    def desenhar_deteccoes_apenas(self, frame, objetos_detectados):
        """
        Apenas desenha as detecções no frame, sem fazer nova detecção.
        Útil para quando a detecção já foi feita em outro lugar.
        """
        try:
            frame_processado = frame.copy()
            
            # Aqui você precisaria ter as coordenadas das detecções
            # Como não temos, retornamos o frame original
            # Em uma implementação completa, você guardaria as últimas coordenadas
            
            return frame_processado
            
        except Exception as e:
            logger.error(f"YOLO: Erro ao desenhar detecções: {e}")
            return frame

    def get_contagem_acumulada(self):
        """Retorna a contagem acumulada atual"""
        return self.contagem_acumulada.copy()

    def get_contagem_pessoas(self):
        """Retorna contagem específica de pessoas"""
        return self.contagem_acumulada.get('person', 0)

    def get_estatisticas(self):
        """Retorna estatísticas completas do detector"""
        objetos_ativos = []
        for obj in self.contagem_acumulada:
            if self.historico_deteccoes[obj]:
                frames_desde_ultima = self.frame_count - max(self.historico_deteccoes[obj])
                objetos_ativos.append(f"{obj}({self.contagem_acumulada[obj]}, {frames_desde_ultima}f)")
        
        return {
            "total_frames": self.frame_count,
            "contagem_acumulada": dict(self.contagem_acumulada),
            "pessoas_detectadas": self.get_contagem_pessoas(),
            "total_objetos": sum(self.contagem_acumulada.values()),
            "objetos_ativos": objetos_ativos
        }

    def resetar_contagem(self):
        """Reseta toda a contagem acumulada"""
        self.contagem_acumulada.clear()
        self.historico_deteccoes.clear()
        self.ultima_deteccao.clear()
        self.frame_count = 0
        self.ultimo_log = 0
        logger.info("🔄 YOLO: Contagem acumulada resetada")