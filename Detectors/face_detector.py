import face_recognition
import cv2
import logging
import numpy as np
from collections import Counter
import time

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, tolerance=0.5):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        self.min_confidence = 0.5
        
        # ✅ NOVO: Controle de logs para performance
        self.ultimo_log_terminal = 0
        self.log_interval = 5  # Segundos entre logs
        
        logger.info("Detector de faces inicializado com sucesso.")

    def carregar_encodings(self, encodings, names):
        """Carrega encodings conhecidos com validação"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Filtrar apenas encodings válidos e nomes não "Desconhecido"
        for encoding, name in zip(encodings, names):
            if (encoding is not None and 
                len(encoding) == 128 and 
                name.strip().lower() not in ['desconhecido', 'unknown', '']):
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name.strip())
        
        logger.info(f"Carregados {len(self.known_face_encodings)} encodings de faces válidos.")
        
        if self.known_face_encodings:
            logger.info(f"Nomes cadastrados: {list(set(self.known_face_names))}")

    def detectar_faces(self, frame):
        """
        Detecta e reconhece faces no frame.
        Retorna: (frame_com_rostos, count, locations, names)
        """
        try:
            # Converter BGR para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encontrar todas as faces no frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            face_names = []
            frame_com_rostos = frame.copy()
            
            # Se não há faces, retornar vazio
            if not face_locations:
                return frame_com_rostos, 0, [], []
            
            # Processar cada face encontrada
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Desconhecido"
                confidence = 0.0
                
                if self.known_face_encodings:
                    # Calcular distâncias para todas as faces conhecidas
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    # Encontrar a melhor correspondência
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    
                    # Converter distância em confiança (0-1)
                    confidence = max(0, 1 - best_distance)
                    
                    # Só considerar reconhecido se estiver dentro da tolerância E com confiança mínima
                    if best_distance <= self.tolerance and confidence >= self.min_confidence:
                        name = self.known_face_names[best_match_index]
                        # ✅ LOG OTIMIZADO: Removido log debug para cada reconhecimento
            
                face_names.append(name)
                
                # Desenhar caixa e nome com cores diferentes
                if name != "Desconhecido":
                    # Conhecido - Verde
                    cor = (0, 255, 0)
                    texto = f"{name} ({confidence:.0%})"
                else:
                    # Desconhecido - Vermelho
                    cor = (0, 0, 255)
                    texto = "Desconhecido"
                
                # Caixa ao redor do rosto
                cv2.rectangle(frame_com_rostos, (left, top), (right, bottom), cor, 2)
                
                # Rótulo com nome
                cv2.rectangle(frame_com_rostos, (left, bottom - 35), (right, bottom), cor, cv2.FILLED)
                cv2.putText(frame_com_rostos, texto, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            # ✅ LOG OTIMIZADO: Análise apenas a cada 5 segundos
            self._analisar_resultados_otimizado(face_names)
            
            return frame_com_rostos, len(face_names), face_locations, face_names
            
        except Exception as e:
            # ✅ LOG OTIMIZADO: Erros apenas a cada 5 segundos
            tempo_atual = time.time()
            if tempo_atual - self.ultimo_log_terminal >= self.log_interval:
                logger.error(f"Erro ao detectar faces: {e}")
                self.ultimo_log_terminal = tempo_atual
            return frame, 0, [], []

    def _analisar_resultados(self, face_names):
        """Faz análise detalhada dos resultados de reconhecimento"""
        if not face_names:
            return
        
        contador = Counter(face_names)
        total_faces = len(face_names)
        conhecidos = {nome: count for nome, count in contador.items() if nome != "Desconhecido"}
        desconhecidos = contador.get("Desconhecido", 0)
        
        # Log informativo
        if conhecidos:
            nomes_conhecidos = ", ".join([f"{nome}({count})" for nome, count in conhecidos.items()])
            logger.info(f"👥 Faces: {total_faces} total | ✅ Conhecidas: {nomes_conhecidos} | ❌ Desconhecidas: {desconhecidos}")
        else:
            logger.info(f"👥 Faces: {total_faces} pessoas (todas não reconhecidas)")

    def _analisar_resultados_otimizado(self, face_names):
        """
        ✅ NOVO: Versão otimizada que só loga a cada 5 segundos
        """
        if not face_names:
            return
            
        tempo_atual = time.time()
        
        # Só logar a cada 5 segundos
        if tempo_atual - self.ultimo_log_terminal >= self.log_interval:
            contador = Counter(face_names)
            total_faces = len(face_names)
            conhecidos = {nome: count for nome, count in contador.items() if nome != "Desconhecido"}
            desconhecidos = contador.get("Desconhecido", 0)
            
            # Log informativo
            if conhecidos:
                nomes_conhecidos = ", ".join([f"{nome}({count})" for nome, count in conhecidos.items()])
                logger.info(f"👥 Faces: {total_faces} total | ✅ Conhecidas: {nomes_conhecidos} | ❌ Desconhecidas: {desconhecidos}")
            else:
                logger.info(f"👥 Faces: {total_faces} pessoas (todas não reconhecidas)")
            
            self.ultimo_log_terminal = tempo_atual

    def detectar_faces_para_interpreter(self, frame):
        """
        Versão otimizada para o interpreter - retorna apenas os nomes
        Retorna: lista de nomes das pessoas detectadas
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if not face_locations:
                return []
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_names = []
            
            for face_encoding in face_encodings:
                name = "Desconhecido"
                
                if self.known_face_encodings:
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    
                    if best_distance <= self.tolerance:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
            
            # ✅ LOG OTIMIZADO: Removido log debug para cada frame
            return face_names
            
        except Exception as e:
            # ✅ LOG OTIMIZADO: Erros apenas a cada 5 segundos
            tempo_atual = time.time()
            if tempo_atual - self.ultimo_log_terminal >= self.log_interval:
                logger.error(f"Erro na detecção para interpreter: {e}")
                self.ultimo_log_terminal = tempo_atual
            return []

    def verificar_encoding_qualidade(self, encoding):
        """
        Verifica a qualidade de um encoding facial
        """
        if encoding is None or len(encoding) != 128:
            return False, "Encoding inválido"
        
        # Verificar se o encoding não é apenas zeros
        if np.all(encoding == 0):
            return False, "Encoding vazio"
        
        # Verificar variância (encoding muito uniforme pode ser ruim)
        variance = np.var(encoding)
        if variance < 0.001:
            return False, f"Encoding com variância muito baixa: {variance:.4f}"
        
        return True, f"Encoding válido (variância: {variance:.4f})"

    def get_nomes_cadastrados(self):
        """Retorna lista de nomes cadastrados"""
        return list(set(self.known_face_names))

    def get_estatisticas(self):
        """Retorna estatísticas do detector"""
        return {
            "total_encodings": len(self.known_face_encodings),
            "nomes_unicos": len(set(self.known_face_names)),
            "tolerance": self.tolerance,
            "min_confidence": self.min_confidence
        }