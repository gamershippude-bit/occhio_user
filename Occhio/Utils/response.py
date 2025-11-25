import logging
import os
import sys

logger = logging.getLogger(__name__)

# Tenta importar TTS da OpenAI
try:
    from Utils.tts_openai import OpenAITTS
    TTS_DISPONIVEL = True
    logger.info("✅ OpenAI TTS disponível")
except ImportError as e:
    TTS_DISPONIVEL = False
    logger.warning(f"⚠️ OpenAI TTS não disponível: {e}")

# Fallback para gTTS
try:
    from gtts import gTTS
    import pygame
    import tempfile
    HAS_GTTS = True
    logger.info("✅ gTTS disponível como fallback")
except ImportError:
    HAS_GTTS = False
    logger.warning("⚠️ gTTS não disponível")

def falar(texto, api_key=None, use_openai=True):
    """
    Fala o texto usando OpenAI TTS (prioridade) ou gTTS (fallback)
    """
    if not texto or not texto.strip():
        logger.warning("Texto vazio para TTS")
        return
    
    logger.info(f"🎤 Falando: {texto}")
    
    # Tentar OpenAI TTS primeiro
    if use_openai and api_key and TTS_DISPONIVEL:
        try:
            tts = OpenAITTS(api_key=api_key)
            tts.falar(texto)
            logger.info("✅ Áudio OpenAI TTS enviado para reprodução")
            return
        except Exception as e:
            logger.error(f"❌ Erro OpenAI TTS, usando fallback: {e}")
    
    # Fallback para gTTS
    if HAS_GTTS:
        try:
            _falar_gtts(texto)
            logger.info("✅ Áudio gTTS reproduzido")
        except Exception as e:
            logger.error(f"❌ Erro gTTS: {e}")
    else:
        logger.error("❌ Nenhum sistema TTS disponível")

def _falar_gtts(texto):
    """Fallback usando gTTS"""
    try:
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Gerar áudio
        tts = gTTS(text=texto, lang='pt-br')
        tts.save(temp_path)
        
        # Reproduzir
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Esperar terminar
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Limpar
        pygame.mixer.quit()
        os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"Erro gTTS: {e}")
        raise