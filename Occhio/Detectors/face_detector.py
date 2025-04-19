"""
Módulo de Detecção e Reconhecimento Facial
Este módulo implementa a detecção e reconhecimento de faces utilizando
a biblioteca face_recognition, fornecendo funcionalidades para:
- Detecção de faces em frames
- Reconhecimento facial baseado em encodings conhecidos
- Contagem e identificação de faces
- Visualização dos resultados
"""

import cv2
import face_recognition
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        try:
            self.encodings_conhecidos = []
            self.nomes_conhecidos = []
            self.ultima_detecao = None
            self.ultima_detecao_time = 0
            self.detecao_interval = 0.5  # Reduzindo intervalo para detecções mais frequentes
            logger.info("Detector de faces inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar detector de faces: {e}")
            raise

    def processar_faces(self, frame, known_face_encodings):
        try:
            # Reduz tamanho do frame para processamento
            height, width = frame.shape[:2]
            small_frame = cv2.resize(frame, (width//2, height//2))
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detecta faces
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Escala as coordenadas de volta
            face_locations = [(top*2, right*2, bottom*2, left*2) 
                            for (top, right, bottom, left) in face_locations]
            
            return frame, face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Erro ao processar faces: {e}")
            return frame, [], []

    def verificar_rosto_existente(self, face_encoding, known_face_encodings):
        try:
            if not known_face_encodings:
                return False
                
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            return any(matches)
            
        except Exception as e:
            logger.error(f"Erro ao verificar rosto existente: {e}")
            return False

    def carregar_encodings(self, encodings, nomes):
        try:
            self.encodings_conhecidos = encodings
            self.nomes_conhecidos = nomes
            logger.info(f"Carregados {len(encodings)} encodings de faces")
        except Exception as e:
            logger.error(f"Erro ao carregar encodings: {e}")

    def detectar_faces(self, frame):
        """Detecta faces em um frame"""
        try:
            # Converte para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Calcula encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Desenha retângulos e adiciona labels
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Desenha retângulo
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Verifica se é uma face conhecida
                matches = face_recognition.compare_faces(self.encodings_conhecidos, face_encoding)
                nome = "Desconhecido"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    nome = self.nomes_conhecidos[first_match_index]
                
                # Adiciona label
                cv2.putText(frame, nome, (left, top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Atualiza atributos
            self.face_locations = face_locations
            self.face_encodings = list(zip(face_locations, face_encodings))
            
            return frame, len(face_locations), self.face_encodings
            
        except Exception as e:
            logger.error(f"Erro ao detectar faces: {e}")
            return frame, 0, []

    def _desenhar_face(self, frame, coords, name):
        """
        Desenha a caixa delimitadora e nome da face no frame.
        
        Args:
            frame: Frame onde desenhar
            coords: Coordenadas da face (top, right, bottom, left)
            name: Nome da pessoa identificada
        """
        top, right, bottom, left = coords
        
        # Desenha retângulo ao redor da face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Desenha fundo para o nome
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 0, 255),
            cv2.FILLED
        )
        
        # Desenha o nome
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1
        )

# Instância global do detector
detector = FaceDetector()

def processar_faces(frame):
    """
    Função de conveniência para processamento de faces.
    
    Args:
        frame: Frame para processamento
        
    Returns:
        Resultados do processamento
    """
    return detector.detectar_faces(frame)
