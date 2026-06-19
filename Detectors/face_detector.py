import face_recognition
import cv2
import logging
import numpy as np
import threading
from collections import Counter
import time

logger = logging.getLogger(__name__)

_DLIB_LOCK = threading.Lock()

class FaceDetector:
    def __init__(self, tolerance=0.5):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        self.min_confidence = 0.5
        self.ultimo_log_terminal = 0
        self.log_interval = 5  # Segundos entre logs
        
        logger.info("Detector de faces inicializado com sucesso.")

    def carregar_encodings(self, encodings, names):
        """Carrega encodings conhecidos — um encoding por nome (evita duplicatas do banco)."""
        with _DLIB_LOCK:
            self.known_face_encodings = []
            self.known_face_names = []
            vistos = set()

            for encoding, name in zip(encodings, names):
                if (encoding is None or len(encoding) != 128):
                    continue
                nome = name.strip()
                if nome.lower() in ('desconhecido', 'unknown', ''):
                    continue
                chave = nome.lower()
                if chave in vistos:
                    continue
                vistos.add(chave)
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(nome)

            logger.info(f"Carregados {len(self.known_face_encodings)} rosto(s) único(s).")
            if self.known_face_names:
                logger.info(f"Nomes cadastrados: {self.known_face_names}")

    def detectar_faces(self, frame):
        """
        Detecta e reconhece faces no frame.
        Retorna: (frame_com_rostos, count, locations, names)
        """
        try:
            with _DLIB_LOCK:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                face_names = []
                frame_com_rostos = frame.copy()
                
                if not face_locations:
                    return frame_com_rostos, 0, [], []
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Desconhecido"
                    confidence = 0.0
                    
                    if self.known_face_encodings:
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, 
                            face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        confidence = max(0, 1 - best_distance)
                        
                        if best_distance <= self.tolerance and confidence >= self.min_confidence:
                            name = self.known_face_names[best_match_index]
                    
                    face_names.append(name)
                    
                    if name != "Desconhecido":
                        cor = (0, 255, 0)
                        texto = f"{name} ({confidence:.0%})"
                    else:
                        cor = (0, 0, 255)
                        texto = "Desconhecido"
                    
                    cv2.rectangle(frame_com_rostos, (left, top), (right, bottom), cor, 2)
                    cv2.rectangle(frame_com_rostos, (left, bottom - 35), (right, bottom), cor, cv2.FILLED)
                    cv2.putText(frame_com_rostos, texto, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
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
            with _DLIB_LOCK:
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

    def contar_rostos(self, frame) -> int:
        """Conta quantos rostos aparecem no frame."""
        try:
            with _DLIB_LOCK:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb, model='hog')
                return len(locations)
        except Exception as e:
            logger.error(f'Erro ao contar rostos: {e}')
            return 0

    def extrair_encoding_principal(self, frame):
        """Extrai encoding do maior rosto no frame."""
        try:
            with _DLIB_LOCK:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb, model='hog')
                if not locations:
                    return None, 'Nenhum rosto detectado. Posicione a pessoa de frente para a câmera.'
                if len(locations) > 1:
                    return None, (
                        f'Estou vendo {len(locations)} rostos na câmera. '
                        'Enquadre apenas uma pessoa para cadastrar.'
                    )

                def area(loc):
                    top, right, bottom, left = loc
                    return (bottom - top) * (right - left)

                locations = sorted(locations, key=area, reverse=True)
                encodings = face_recognition.face_encodings(rgb, [locations[0]])
                if not encodings:
                    return None, 'Não consegui capturar o rosto com qualidade suficiente.'

                encoding = encodings[0]
                ok, msg = self.verificar_encoding_qualidade(encoding)
                if not ok:
                    return None, msg
                return encoding, None
        except Exception as e:
            logger.error(f'Erro ao extrair encoding: {e}')
            return None, 'Erro ao processar o rosto.'

    @staticmethod
    def _iou_box(a, b):
        """IoU entre duas caixas normalizadas {x, y, w, h}."""
        ax2, ay2 = a['x'] + a['w'], a['y'] + a['h']
        bx2, by2 = b['x'] + b['w'], b['y'] + b['h']
        ix1, iy1 = max(a['x'], b['x']), max(a['y'], b['y'])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = a['w'] * a['h']
        area_b = b['w'] * b['h']
        return inter / (area_a + area_b - inter)

    def _deduplicar_rostos(self, rostos):
        """Remove caixas sobrepostas e agrupa rostos conhecidos pelo mesmo nome."""
        if len(rostos) <= 1:
            return rostos

        rostos = sorted(rostos, key=lambda r: r['w'] * r['h'], reverse=True)
        unicos = []
        for rosto in rostos:
            if any(self._iou_box(rosto, u) > 0.4 for u in unicos):
                continue
            unicos.append(rosto)

        por_nome = {}
        desconhecidos = []
        for r in unicos:
            if r.get('conhecido') and r.get('nome'):
                chave = r['nome'].lower()
                if chave not in por_nome or r['confianca'] > por_nome[chave]['confianca']:
                    por_nome[chave] = r
            else:
                desconhecidos.append(r)

        return list(por_nome.values()) + desconhecidos

    def detectar_faces_bbox(self, frame):
        """Detecta rostos e retorna bbox normalizadas para o canvas."""
        try:
            with _DLIB_LOCK:
                altura, largura = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb, model='hog')
                if not locations:
                    return []

                encodings = face_recognition.face_encodings(rgb, locations)
                resultado = []
                area_min = 0.008 * largura * altura

                for (top, right, bottom, left), face_encoding in zip(locations, encodings):
                    w = right - left
                    h = bottom - top
                    if w * h < area_min:
                        continue

                    nome = 'Desconhecido'
                    confianca = 0.0

                    if self.known_face_encodings:
                        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best = int(np.argmin(distances))
                        dist = float(distances[best])
                        confianca = max(0.0, 1.0 - dist)
                        if dist <= self.tolerance and confianca >= self.min_confidence:
                            nome = self.known_face_names[best]

                    resultado.append({
                        'nome': nome,
                        'confianca': round(confianca, 2),
                        'conhecido': nome != 'Desconhecido',
                        'x': round(left / largura, 4),
                        'y': round(top / altura, 4),
                        'w': round(w / largura, 4),
                        'h': round(h / altura, 4),
                    })

                return self._deduplicar_rostos(resultado)
        except Exception as e:
            logger.error(f'Erro em detectar_faces_bbox: {e}')
            return []