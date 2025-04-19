# occhio/report/pdf_report.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class PDFReport:
    def __init__(self):
        """Inicializa o relatório PDF"""
        try:
            # Cria diretório se não existir
            os.makedirs("relatorios", exist_ok=True)
            
            # Cria nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"relatorios/relatorio_{timestamp}.pdf"
            
            # Inicializa o canvas
            self.c = canvas.Canvas(self.filename, pagesize=letter)
            self.y_position = 750  # Posição inicial Y
            self._add_header()
            
            logger.info("Relatório PDF inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar PDF: {e}")
            self.c = None

    def _add_header(self):
        """Adiciona cabeçalho ao relatório"""
        if self.c:
            self.c.setFont("Helvetica-Bold", 16)
            self.c.drawString(50, self.y_position, "Relatório de Detecção")
            self.c.setFont("Helvetica", 12)
            self.c.drawString(50, self.y_position - 20, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.y_position -= 50

    def add_face(self, face_path, face_id, face_location):
        """Adiciona uma face detectada ao relatório"""
        try:
            if self.c:
                # Adiciona imagem da face
                self.c.drawImage(face_path, 50, self.y_position - 160, width=160, height=160)
                
                # Adiciona informações
                self.c.setFont("Helvetica", 10)
                self.c.drawString(220, self.y_position - 80, f"Rosto ID: {face_id}")
                self.c.drawString(220, self.y_position - 100, f"Localização: {face_location}")
                self.c.drawString(220, self.y_position - 120, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.y_position -= 180
                
                # Nova página se necessário
                if self.y_position < 100:
                    self._new_page()
                    
        except Exception as e:
            logger.error(f"Erro ao adicionar face ao PDF: {e}")

    def add_detection_summary(self, objetos_detectados, faces_detectadas):
        """Adiciona um resumo das detecções ao relatório"""
        try:
            if self.c:
                self.c.setFont("Helvetica-Bold", 12)
                self.c.drawString(50, self.y_position, "Resumo de Detecções")
                self.y_position -= 20
                
                self.c.setFont("Helvetica", 10)
                if objetos_detectados:
                    self.c.drawString(50, self.y_position, "Objetos Detectados:")
                    self.y_position -= 15
                    for obj in objetos_detectados:
                        self.c.drawString(70, self.y_position, f"- {obj}")
                        self.y_position -= 15
                
                if faces_detectadas:
                    self.c.drawString(50, self.y_position, "Faces Detectadas:")
                    self.y_position -= 15
                    for face in faces_detectadas:
                        self.c.drawString(70, self.y_position, f"- {face}")
                        self.y_position -= 15
                
                self.y_position -= 20
                
                # Nova página se necessário
                if self.y_position < 100:
                    self._new_page()
                    
        except Exception as e:
            logger.error(f"Erro ao adicionar resumo ao PDF: {e}")

    def _new_page(self):
        """Cria uma nova página no relatório"""
        if self.c:
            self.c.showPage()
            self.y_position = 750
            self._add_header()

    def save(self):
        """Salva o relatório PDF"""
        try:
            if self.c:
                self.c.save()
                logger.info(f"Relatório salvo em: {self.filename}")
        except Exception as e:
            logger.error(f"Erro ao salvar PDF: {e}")

# Instância global do relatório
report = PDFReport()

def criar_pdf():
    pdf_filename = "relatorio_deteccao.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "Relatório de Detecção de Objetos e Rostos")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "------------------------------------")
    y_position = 710
    return c, y_position

def adicionar_rosto_pdf(c, y_position, face_id, face_filename):
    try:
        c.drawString(100, y_position, f"Rosto detectado {face_id} salvo")
        y_position -= 20
        if y_position < 150:
            c.showPage()
            y_position = 750

        c.drawImage(face_filename, 100, y_position - 100, width=100, height=100)
        y_position -= 120
        return y_position
    except Exception as e:
        logger.error(f"Erro ao adicionar rosto ao PDF: {str(e)}")
        return y_position

def adicionar_deteccoes_pdf(c, y_position, frame_count, fps, contagem_objetos, num_faces):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position, f"Tempo {frame_count // fps:.1f}s:")
    y_position -= 20
    c.setFont("Helvetica", 12)

    for obj, qtd in contagem_objetos.items():
        c.drawString(120, y_position, f"{obj.capitalize()}: {qtd}")
        y_position -= 20

    if num_faces > 0:
        c.drawString(120, y_position, f"Rostos detectados: {num_faces}")
        y_position -= 20

    if y_position < 100:
        c.showPage()
        y_position = 750

    return y_position

def adicionar_resumo_pdf(c, y_position, contagem_total, total_faces_detectados, objetos_detectados):
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "Resumo da Detecção")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "------------------------------------")
    y_position = 710

    if objetos_detectados:
        for obj, qtd in contagem_total.items():
            c.drawString(100, y_position, f"{obj.capitalize()}: {qtd} vezes")
            y_position -= 20

        c.drawString(100, y_position, f"Total de Rostos Detectados: {total_faces_detectados} vezes")
    else:
        c.drawString(100, y_position, "Nenhum objeto ou rosto foi detectado.")

    c.save()
    logger.info("Relatório PDF salvo com sucesso")
