# Occhio - Sistema de Visão Computacional

Sistema de visão computacional desenvolvido para detecção de objetos e faces, com suporte a câmeras locais e ESP.

## Funcionalidades

- Detecção de objetos usando YOLO
- Detecção e reconhecimento facial
- Suporte a câmeras locais e ESP
- Geração de relatórios em PDF
- Armazenamento de dados em banco SQLite

## Requisitos

- Python 3.8+
- OpenCV
- NumPy
- face_recognition
- ultralytics (YOLO)
- reportlab
- requests
- Pillow

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/occhio.git
cd occhio
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Câmera Local
```bash
python Occhio/Main.py --source 0
```

### Câmera ESP
```bash
python Occhio/Main.py --source "192.168.1.101"
```

## Estrutura do Projeto

```
Occhio/
├── Main.py              # Ponto de entrada principal
├── Detectors/           # Detectores de objetos e faces
├── Utils/              # Utilitários e geradores
├── db/                 # Banco de dados
└── Report/             # Geração de relatórios
```

## Contribuição

Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes. 