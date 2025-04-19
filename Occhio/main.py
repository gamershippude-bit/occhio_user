"""
Occhio - Sistema de Visão Computacional
Este módulo é o ponto de entrada principal do sistema, responsável por:
- Inicialização da câmera
- Processamento de frames
- Detecção de objetos e faces
- Geração de descrições
- Interface com o usuário
"""

import cv2
import argparse
import logging
import time
from datetime import datetime
import os
from Detectors.yolo_detector import YOLODetector
from Detectors.face_detector import FaceDetector
from Utils.gerador import GeradorDescricao
from db.database import conectar_db, carregar_encodings_existentes, salvar_rosto
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from Report.pdf_report import PDFReport
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,  # Aumentando nível de log para debug
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ESPCamera:
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.stream = None
        
    def read(self):
        try:
            response = self.session.get(self.url, stream=True, timeout=5)
            if response.status_code == 200:
                # Log do tipo de conteúdo recebido
                content_type = response.headers.get('Content-Type', '')
                logger.info(f"Content-Type recebido: {content_type}")
                
                # Tenta diferentes métodos de decodificação
                try:
                    # Método 1: Diretamente com PIL
                    img = Image.open(BytesIO(response.content))
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    return True, frame
                except Exception as e1:
                    logger.warning(f"Falha método 1: {e1}")
                    try:
                        # Método 2: Usando numpy diretamente
                        nparr = np.frombuffer(response.content, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            return True, frame
                    except Exception as e2:
                        logger.warning(f"Falha método 2: {e2}")
                        try:
                            # Método 3: Salvando temporariamente
                            temp_path = "temp_frame.jpg"
                            with open(temp_path, 'wb') as f:
                                f.write(response.content)
                            frame = cv2.imread(temp_path)
                            if frame is not None:
                                return True, frame
                        except Exception as e3:
                            logger.warning(f"Falha método 3: {e3}")
                            
                logger.error("Todos os métodos de decodificação falharam")
                return False, None
            else:
                logger.error(f"Status code inválido: {response.status_code}")
                return False, None
        except Exception as e:
            logger.error(f"Erro ao ler frame da câmera ESP: {e}")
            return False, None
            
    def isOpened(self):
        try:
            response = self.session.get(self.url, timeout=5)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                logger.info(f"Conexão estabelecida. Content-Type: {content_type}")
                return True
            logger.error(f"Status code inválido: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar conexão: {e}")
            return False
            
    def release(self):
        if self.session:
            self.session.close()

class Occhio:
    def __init__(self, source=0, width=640, height=480):  # Aumentando resolução para melhor detecção
        """
        Inicializa o sistema Occhio.
        
        Args:
            source (int/str): Fonte de vídeo (0 para webcam local, URL para câmera IP)
            width (int): Largura do frame
            height (int): Altura do frame
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.detector_objetos = YOLODetector()
        self.detector_faces = FaceDetector()
        self.gerador = GeradorDescricao()
        self.ultima_descricao = ""
        self.ultima_descricao_time = 0
        self.descricao_interval = 5
        self.ultimo_frame_time = 0
        self.frame_interval = 1/30
        self.ultimo_frame = None
        self.ultima_detecao = None
        self.ultima_detecao_time = 0
        self.detecao_interval = 0.5  # Reduzindo intervalo entre detecções
        self.ultimo_salvamento = 0
        self.salvamento_interval = 1
        self.rostos_salvos = set()  # IDs dos rostos já salvos
        self.encodings_salvos = []  # Lista de encodings já salvos
        self.c = None
        self.y_position = 750
        self.pdf_report = None
        self.setup_database()
        self.setup_pdf()
        self.setup_diretorios()

    def setup_diretorios(self):
        """Cria diretórios necessários para salvamento"""
        try:
            os.makedirs("rostos", exist_ok=True)
            os.makedirs("relatorios", exist_ok=True)
            logger.info("Diretórios criados com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar diretórios: {e}")

    def setup_database(self):
        """Configura a conexão com o banco de dados e carrega encodings existentes"""
        try:
            self.conn, self.cursor = conectar_db()
            if self.conn and self.cursor:
                # Carrega encodings existentes do banco
                encodings, nomes = carregar_encodings_existentes(self.cursor)
                self.encodings_salvos = encodings  # Inicializa com encodings do banco
                logger.info(f"Carregados {len(encodings)} encodings do banco de dados")
        except Exception as e:
            logger.error(f"Erro ao configurar banco de dados: {e}")
            self.conn = None
            self.cursor = None
            self.encodings_salvos = []

    def setup_pdf(self):
        """Configura o arquivo PDF para relatórios"""
        try:
            self.pdf_report = PDFReport()
            logger.info("PDF configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar PDF: {e}")
            self.pdf_report = None

    def process_frame(self, frame):
        """Processa um frame da câmera"""
        try:
            # Detecta objetos
            frame, contagem_objetos, objetos_detectados = self.detector_objetos.detectar_objetos(frame)
            
            # Detecta faces
            frame, contagem_faces, faces_detectadas = self.detector_faces.detectar_faces(frame)
            
            # Converte contagens para inteiros se necessário
            contagem_objetos = int(contagem_objetos) if isinstance(contagem_objetos, (int, float)) else 0
            contagem_faces = int(contagem_faces) if isinstance(contagem_faces, (int, float)) else 0
            
            # Atualiza o PDF com o resumo das detecções
            if self.pdf_report and (contagem_objetos > 0 or contagem_faces > 0):
                try:
                    self.pdf_report.add_detection_summary(objetos_detectados, faces_detectadas)
                except Exception as e:
                    logger.error(f"Erro ao adicionar resumo ao PDF: {e}")
            
            return frame, contagem_objetos, contagem_faces
            
        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}")
            return frame, 0, 0

    def gerar_descricao(self, detecao):
        """
        Gera uma descrição textual baseada nas detecções atuais.
        
        Args:
            detecao: Dicionário contendo informações de detecção
            
        Returns:
            String com a descrição gerada
        """
        current_time = time.time()
        
        # Verifica se é hora de gerar nova descrição
        if (current_time - self.ultima_descricao_time < self.descricao_interval and 
            self.ultima_descricao):
            return self.ultima_descricao
            
        try:
            descricao = self.gerador.gerar_descricao(detecao)
            self.ultima_descricao = descricao
            self.ultima_descricao_time = current_time
            return descricao
            
        except Exception as e:
            logger.error(f"Erro ao gerar descrição: {e}")
            return ""

    def _e_distancia(self, encoding1, encoding2):
        """Calcula a distância euclidiana entre dois encodings"""
        return np.linalg.norm(encoding1 - encoding2)

    def _e_rosto_ja_salvo(self, novo_encoding):
        """Verifica se o rosto já foi salvo comparando com encodings existentes"""
        if not self.encodings_salvos:
            return False
            
        for encoding_salvo in self.encodings_salvos:
            try:
                distancia = self._e_distancia(novo_encoding, encoding_salvo)
                if distancia < 0.6:  # Limiar de similaridade
                    logger.info(f"Rosto similar encontrado (distância: {distancia:.2f})")
                    return True
            except Exception as e:
                logger.warning(f"Erro ao comparar encodings: {e}")
                continue
        return False

    def salvar_rosto_detectado(self, frame, face_location, face_encoding):
        current_time = time.time()
        
        if current_time - self.ultimo_salvamento < self.salvamento_interval:
            return
            
        try:
            # Extrai coordenadas do rosto
            if isinstance(face_location, tuple) and len(face_location) == 4:
                top, right, bottom, left = face_location
            else:
                logger.error("Formato inválido de face_location")
                return
                
            # Verifica se o rosto já foi salvo comparando encodings
            if self._e_rosto_ja_salvo(face_encoding):
                logger.info("Rosto já existe no banco de dados (encoding similar encontrado)")
                return
                
            # Gera um ID único baseado no timestamp
            face_id = f"{int(current_time)}"
            
            # Cria diretório se não existir
            os.makedirs("rostos", exist_ok=True)
            
            # Salva imagem do rosto
            face_img = frame[top:bottom, left:right]
            face_img = cv2.resize(face_img, (160, 160))  # Redimensiona para tamanho padrão
            face_path = f"rostos/rosto_{face_id}.jpg"
            cv2.imwrite(face_path, face_img)
            
            if self.conn and self.cursor and self.pdf_report:
                # Salva no banco de dados
                nome_rosto = f"Rosto_{face_id}"
                if salvar_rosto(face_id, face_encoding, nome_rosto, self.cursor, self.conn):
                    # Adiciona ao PDF
                    self.pdf_report.add_face(face_path, face_id, face_location)
                    
                    # Adiciona aos conjuntos de rostos salvos
                    self.rostos_salvos.add(face_id)
                    self.encodings_salvos.append(face_encoding)
                    self.ultimo_salvamento = current_time
                    logger.info(f"Rosto salvo com sucesso: {face_id}")
                else:
                    logger.error(f"Falha ao salvar rosto {face_id} no banco de dados")
                    
        except Exception as e:
            logger.error(f"Erro ao salvar rosto: {e}")

    def run(self):
        """Loop principal do sistema"""
        self.cap = self.create_capture()
        if not self.cap:
            logger.error("Não foi possível inicializar a câmera")
            return
            
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame vazio recebido")
                    continue

                # Processa o frame
                frame, contagem_objetos, contagem_faces = self.process_frame(frame)
                
                # Gera e exibe descrição
                descricao = self.gerar_descricao({
                    "objetos": contagem_objetos,
                    "faces": contagem_faces
                })
                
                if descricao:
                    cv2.putText(frame, descricao, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Exibe o frame
                cv2.imshow('Occhio', frame)
                
                # Verifica comandos do usuário
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Salva rosto quando tecla 's' é pressionada
                    if contagem_faces > 0 and hasattr(self.detector_faces, 'face_encodings'):
                        for face_location, face_encoding in self.detector_faces.face_encodings:
                            self.salvar_rosto_detectado(frame, face_location, face_encoding)
                
        except Exception as e:
            logger.error(f"Erro durante execução: {e}")
            
        finally:
            self.cleanup()

    def cleanup(self):
        """Libera recursos e finaliza o sistema"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.conn:
            self.conn.close()
        if self.pdf_report:
            self.pdf_report.save()
        logger.info("Sistema finalizado")

    def create_capture(self):
        """
        Inicializa a captura de vídeo.
        Tenta diferentes métodos para garantir compatibilidade com diferentes câmeras.
        """
        try:
            # Se source for uma URL (não um número)
            if isinstance(self.source, str) and not self.source.isdigit():
                # Remove http:// se existir para evitar duplicação
                base_url = self.source.replace('http://', '')
                
                # Tenta diferentes formatos de URL comuns para câmeras ESP
                urls = [
                    f"http://{base_url}",  # URL original
                    f"http://{base_url}/capture",  # Adiciona /capture
                    f"http://{base_url}/snapshot",  # Adiciona /snapshot
                    f"http://{base_url}/jpg",  # Adiciona /jpg
                    f"http://{base_url}/stream",  # Adiciona /stream
                    f"http://{base_url}/cam.jpg",  # Adiciona /cam.jpg
                ]
                
                for url in urls:
                    try:
                        logger.info(f"Tentando conectar com: {url}")
                        cap = ESPCamera(url)
                        if cap.isOpened():
                            # Tenta ler um frame para confirmar que está funcionando
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                logger.info(f"Câmera inicializada com sucesso: {url}")
                                return cap
                            else:
                                logger.warning(f"Conexão estabelecida mas não foi possível ler frames de {url}")
                                cap.release()
                    except Exception as e:
                        logger.warning(f"Falha ao conectar com {url}: {e}")
                        continue
                
                raise Exception("Não foi possível conectar com nenhum formato de URL. Verifique se:\n"
                              "1. A câmera está ligada e na mesma rede\n"
                              "2. O IP está correto\n"
                              "3. A câmera está transmitindo em um formato compatível")
            else:
                # Para webcam local
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    raise Exception("Não foi possível abrir a câmera")
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                logger.info(f"Câmera inicializada com sucesso: {self.source}")
                return cap
            
        except Exception as e:
            logger.error(f"Erro ao inicializar câmera: {e}")
            return None

def main():
    """Função principal que configura e inicia o sistema"""
    parser = argparse.ArgumentParser(description='Sistema de Visão Computacional Occhio')
    parser.add_argument('--source', type=str, default="0",
                      help='Fonte de vídeo (0 para webcam, URL para câmera IP/ESP)')
    parser.add_argument('--width', type=int, default=640,
                      help='Largura do frame (VGA = 640)')
    parser.add_argument('--height', type=int, default=480,
                      help='Altura do frame (VGA = 480)')
    
    args = parser.parse_args()
    
    try:
        # Se source for um número, converte para int, senão usa como string (URL)
        source = int(args.source) if args.source.isdigit() else args.source
        # Força resolução VGA
        occhio = Occhio(source=source, width=640, height=480)
        occhio.run()
    except Exception as e:
        logger.error(f"Erro ao iniciar sistema: {e}")

if __name__ == "__main__":
    main()
