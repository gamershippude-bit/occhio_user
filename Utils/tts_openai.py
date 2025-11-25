import openai
import logging
import os
import tempfile
import pygame
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class OpenAITTS:
    """Text-to-Speech usando a API da OpenAI"""
    
    def __init__(self, api_key, model="tts-1", voice="alloy"):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.client = openai.OpenAI(api_key=api_key)
        self.is_playing = False
        self.current_file = None
        
        # Inicializar pygame mixer uma vez
        try:
            pygame.mixer.init()
            logger.info("🔊 Pygame mixer inicializado para TTS")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar pygame mixer: {e}")
    
    def gerar_audio(self, texto, output_file=None):
        """
        Gera arquivo de áudio a partir do texto.
        """
        if not texto or not texto.strip():
            logger.warning("⚠️ Texto vazio para TTS")
            return None
            
        try:
            logger.info(f"🔊 Gerando áudio TTS: '{texto[:50]}...'")
            
            # Usar arquivo temporário se não for especificado
            if output_file is None:
                temp_dir = tempfile.gettempdir()
                output_file = os.path.join(temp_dir, f"tts_{hash(texto)}.mp3")
            
            # Chamar API da OpenAI
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=texto
            )
            
            # Salvar arquivo
            response.stream_to_file(output_file)
            logger.info(f"✅ Áudio TTS gerado: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar áudio TTS: {e}")
            return None
    
    def falar(self, texto, wait=False):
        """
        Fala o texto usando TTS (assíncrono por padrão).
        """
        def _falar_thread():
            try:
                self.is_playing = True
                audio_file = self.gerar_audio(texto)
                
                if audio_file and os.path.exists(audio_file):
                    # Reproduzir áudio
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    
                    # Esperar terminar de tocar
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                    
                    # Limpar arquivo temporário
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                    
                self.is_playing = False
                logger.info("✅ Reprodução de áudio concluída")
                
            except Exception as e:
                logger.error(f"❌ Erro na reprodução TTS: {e}")
                self.is_playing = False
        
        if wait:
            _falar_thread()  # Síncrono
        else:
            # Assíncrono em thread separada
            thread = threading.Thread(target=_falar_thread, daemon=True)
            thread.start()
    
    def falar_sincrono(self, texto):
        """Versão síncrona (bloqueante)"""
        self.falar(texto, wait=True)
    
    def esta_falando(self):
        """Verifica se está falando no momento"""
        return self.is_playing or (pygame.mixer.music.get_busy() if pygame.mixer.get_init() else False)
    
    def parar(self):
        """Para a fala atual"""
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
            logger.info("⏹️ Fala interrompida")
        except Exception as e:
            logger.error(f"Erro ao parar fala: {e}")

# Função de conveniência
def criar_tts(api_key):
    """Cria instância do TTS"""
    return OpenAITTS(api_key=api_key)