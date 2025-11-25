# speech_recognition_ultra_otimizado.py
import speech_recognition as sr
import logging
import time
import threading
import re
import pyaudio

# Configuração de logging
logger = logging.getLogger(__name__)

def testar_microfones():
    """Testa e lista os microfones disponíveis"""
    try:
        audio = pyaudio.PyAudio()
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        microfones = []
        for i in range(0, num_devices):
            device_info = audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                microfones.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'rate': int(device_info.get('defaultSampleRate', 16000))
                })
        
        audio.terminate()
        return microfones
    except Exception as e:
        logger.error(f"Erro ao listar microfones: {e}")
        return []

def corrigir_texto_avancado(texto):
    """Correções mais avançadas e contextuais"""
    if not texto:
        return texto
    
    texto = texto.lower().strip()
    
    # Correções básicas de pronúncia
    correcoes_basicas = {
        r'quis ver': 'o que vê',
        r'que vê': 'o que vê', 
        r'quiser': 'o que vê',
        r'que ve': 'o que vê',
        r'quis ve': 'o que vê',
        r'o que é': 'o que vê',
        r'o que eh': 'o que vê',
        r'oxe': 'o que vê',
        r'cadê': 'onde está',
        r'onde esta': 'onde está',
        r'quem é': 'quem está',
        r'quem eh': 'quem está',
        r'quantos': 'quantos tem',
        r'descreve': 'descreva',
        r'discr[ea].*': 'descreva',
        r'descr[ea].*': 'descreva',
        r'^discre$': 'descreva',
        r'^descre$': 'descreva',
        r'discreva': 'descreva',
        r'discreve': 'descreva',
        r'o que você vê': 'o que vê',
        r'o que tu vês': 'o que vê',
        r'o que tá vendo': 'o que vê',
        r'tá vendo': 'o que vê',
        r'me descreva': 'descreva',
        r'pode descrever': 'descreva',
        r'consegue ver': 'o que vê',
        r'tem alguém': 'quem está',
        r'tem pessoas': 'quem está',
        r'reconhece alguém': 'quem está',
        r'quantas pessoas': 'quantos tem',
        r'quantos rostos': 'quantos tem',
    }
    
    # Aplicar correções básicas
    for padrao, correcao in correcoes_basicas.items():
        if re.search(padrao, texto):
            texto_original = texto
            texto = re.sub(padrao, correcao, texto)
            if texto_original != texto:
                logger.info(f"🔧 Correção básica: '{texto_original}' -> '{texto}'")
    
    # Correções contextuais (frases comuns)
    frases_comuns = {
        r'^o que v[êe]': 'o que vê',
        r'^quem est[aá]': 'quem está', 
        r'^onde est[aá]': 'onde está',
        r'^quantos tem': 'quantos tem',
        r'^descreva': 'descreva',
        r'^descreve': 'descreva',
        r'^me descreva': 'descreva',
    }
    
    for padrao, correcao in frases_comuns.items():
        if re.match(padrao, texto):
            texto_original = texto
            texto = correcao
            if texto_original != texto:
                logger.info(f"🔧 Correção contextual: '{texto_original}' -> '{texto}'")
            break
    
    return texto

def encontrar_melhor_microfone():
    """Encontra o melhor microfone disponível"""
    microfones = testar_microfones()
    
    if not microfones:
        logger.warning("❌ Nenhum microfone encontrado")
        return None
    
    logger.info(f"🎙️ Microfones encontrados: {len(microfones)}")
    for mic in microfones:
        logger.info(f"   - {mic['index']}: {mic['name']} ({mic['rate']}Hz)")
    
    # Preferir microfone padrão primeiro
    return None  # None usa o padrão do sistema

def reconhecer_voz_google_ultra_otimizado(timeout=15):
    """
    Google Speech Recognition ULTRA-OTIMIZADO
    """
    recognizer = sr.Recognizer()
    
    # Configurações SUPER OTIMIZADAS
    recognizer.energy_threshold = 250  # Muito sensível
    recognizer.dynamic_energy_threshold = True
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.pause_threshold = 1.2   # Mais tolerante a pausas
    recognizer.phrase_threshold = 0.3  # Detecta frases mais curtas
    recognizer.non_speaking_duration = 0.5
    
    # Tentar diferentes microfones
    microfone_index = encontrar_melhor_microfone()
    
    try:
        if microfone_index is not None:
            with sr.Microphone(device_index=microfone_index) as source:
                return _processar_audio(recognizer, source, timeout)
        else:
            with sr.Microphone() as source:
                return _processar_audio(recognizer, source, timeout)
                
    except Exception as e:
        logger.error(f"❌ Erro grave no microfone: {e}")
        return None

def _processar_audio(recognizer, source, timeout):
    """Processa o áudio de forma otimizada"""
    logger.info("🎤 CALIBRANDO microfone... (aguarde 2 segundos)")
    
    # Calibração mais longa e precisa
    recognizer.adjust_for_ambient_noise(source, duration=2.0)
    
    # Obter nível de ruído atual
    try:
        # Teste rápido do nível de ruído
        recognizer.energy_threshold = recognizer.energy_threshold * 0.8  # Ajuste fino
        logger.info(f"📊 Energy threshold ajustado: {recognizer.energy_threshold}")
    except:
        pass
    
    logger.info("🎧 OUVINDO... FALE AGORA (fale claramente)")
    
    try:
        # Captura de áudio com configurações otimizadas
        audio = recognizer.listen(
            source, 
            timeout=timeout,
            phrase_time_limit=10  # Limite de frase
        )
        
        logger.info("🔄 PROCESSANDO áudio...")
        
        # Tentativas com diferentes configurações
        tentativas = [
            {"language": "pt-BR"},  # Português Brasil
            {"language": "pt-PT"},  # Português Portugal (fallback)
        ]
        
        for config in tentativas:
            try:
                texto = recognizer.recognize_google(audio, **config)
                if texto and texto.strip():
                    logger.info(f"✅ RECONHECIDO: '{texto}'")
                    
                    texto_corrigido = corrigir_texto_avancado(texto)
                    if texto_corrigido != texto:
                        logger.info(f"🔧 TEXTO CORRIGIDO: '{texto_corrigido}'")
                    
                    return texto_corrigido
                    
            except sr.UnknownValueError:
                continue
            except Exception as e:
                logger.warning(f"⚠️ Tentativa falhou: {e}")
                continue
        
        logger.warning("🔇 Não foi possível entender o áudio em nenhuma tentativa")
        return None
        
    except sr.WaitTimeoutError:
        logger.info("⏰ TEMPO ESGOTADO: Ninguém falou")
        return None
    except Exception as e:
        logger.error(f"❌ ERRO na captura: {e}")
        return None

def reconhecer_voz_melhorado(timeout=15):
    """
    Função principal com múltiplas tentativas
    """
    logger.info("🎯 INICIANDO RECONHECIMENTO DE VOZ")
    
    # Tentar até 2 vezes se necessário
    for tentativa in range(2):
        logger.info(f"🔄 Tentativa {tentativa + 1}...")
        
        texto = reconhecer_voz_google_ultra_otimizado(timeout)
        
        if texto:
            return texto
        
        if tentativa == 0:
            logger.info("🔄 Primeira tentativa falhou, tentando novamente...")
            time.sleep(1)  # Pequena pausa entre tentativas
    
    logger.warning("❌ Todas as tentativas falharam")
    return None

class HotwordListener:
    def __init__(self, palavra_chave="ok", callback=None):
        self.palavra_chave = palavra_chave.lower()
        self.callback = callback
        self.ativo = False
        self.recognizer = sr.Recognizer()
        self.microfone = sr.Microphone()
        
        # Configurações para hotword (balance entre sensibilidade e precisão)
        self.recognizer.energy_threshold = 280
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.non_speaking_duration = 0.6
        
        self.listener_thread = None
        logger.info(f"🔥 HOTWORD: '{self.palavra_chave}'")

    def _listen_continuous(self):
        """Escuta contínua para hotword"""
        logger.info("👂 ESCUTANDO hotword...")
        
        with self.microfone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            while self.ativo:
                try:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=0.8,
                        phrase_time_limit=2.0
                    )
                    
                    try:
                        texto = self.recognizer.recognize_google(
                            audio, 
                            language="pt-BR"
                        ).lower()
                        
                        if texto and self.palavra_chave in texto:
                            logger.info(f"🎯 HOTWORD DETECTADA: '{texto}'")
                            if self.callback:
                                self.callback()
                                
                    except sr.UnknownValueError:
                        continue
                    except Exception:
                        continue
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    time.sleep(0.2)

    def start(self):
        if self.ativo:
            return
            
        self.ativo = True
        self.listener_thread = threading.Thread(
            target=self._listen_continuous, 
            daemon=True
        )
        self.listener_thread.start()
        logger.info("🔥 HOTWORD LISTENER INICIADO")

    def stop(self):
        self.ativo = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
        logger.info("🛑 HOTWORD LISTENER PARADO")

# Teste interativo
def teste_interativo():
    """Teste interativo do reconhecimento"""
    print("\n" + "="*60)
    print("🎯 TESTE INTERATIVO DE RECONHECIMENTO DE VOZ")
    print("="*60)
    print("Dicas:")
    print("  • Fale claramente e em volume normal")
    print("  • Mantenha uma distância consistente do microfone") 
    print("  • Evite ruídos de fundo")
    print("  • Frases sugeridas: 'o que vê', 'quem está', 'descreva'")
    print("="*60)
    
    while True:
        input("\nPressione ENTER para começar a escutar (ou 'q' para sair)... ")
        
        print("\n🎤 ESCUTANDO... FALE AGORA!")
        texto = reconhecer_voz_melhorado(timeout=10)
        
        if texto:
            print(f"🎉 SUCESSO: '{texto}'")
            
            # Verificar se quer sair
            if texto.lower() in ['sair', 'parar', 'quit', 'q']:
                break
        else:
            print("❌ Nada reconhecido - tente novamente")
            
        time.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    teste_interativo()