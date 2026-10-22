@echo off
REM ============================================================
REM  Setup Conda - 3GPP RAG Benchmark
REM  Windows + NVIDIA GPU + 8GB VRAM + QLoRA 4-bit
REM  Modeles : Qwen2.5-7B | Llama-3.1-8B | Mistral-7B-v0.3
REM ============================================================

SET ENV_NAME=3gpprag
SET PYTHON_VERSION=3.10

echo.
echo ============================================================
echo  Creation de l'environnement Conda : %ENV_NAME%
echo ============================================================

REM --- Creer l'environnement Conda ---
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if errorlevel 1 (
    echo [ERREUR] Impossible de creer l'environnement Conda.
    pause
    exit /b 1
)

REM --- Activer ---
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Installation PyTorch 2.2.x avec CUDA 11.8
echo  (Compatible 8GB VRAM, Windows, bitsandbytes)
echo ============================================================
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERREUR] Echec installation PyTorch.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Installation Hugging Face + Fine-tuning (QLoRA)
echo ============================================================
pip install ^
    transformers==4.40.2 ^
    accelerate==0.29.3 ^
    peft==0.10.0 ^
    trl==0.8.6 ^
    datasets==2.19.1 ^
    tokenizers==0.19.1 ^
    huggingface-hub==0.22.2 ^
    safetensors==0.4.3

REM bitsandbytes Windows build (obligatoire pour QLoRA 4-bit)
pip install bitsandbytes==0.43.1 --prefer-binary --extra-index-url https://huggingface.github.io/bitsandbytes-windows-webui
if errorlevel 1 (
    echo [AVERTISSEMENT] bitsandbytes via source 1 echoue, tentative source 2...
    pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.3.post2-py3-none-win_amd64.whl
)

echo.
echo ============================================================
echo  Installation RAG Stack
echo  (ChromaDB, LangChain, sentence-transformers, rerankers)
echo ============================================================
pip install ^
    langchain==0.1.20 ^
    langchain-community==0.0.38 ^
    langchain-huggingface==0.0.3 ^
    chromadb==0.4.24 ^
    sentence-transformers==2.7.0 ^
    faiss-cpu==1.8.0 ^
    rank-bm25==0.2.2 ^
    flashrank==0.2.0 ^
    tiktoken==0.6.0 ^
    unstructured==0.13.7 ^
    markdown==3.6

echo.
echo ============================================================
echo  Installation Evaluation & Metrics
echo ============================================================
pip install ^
    rouge-score==0.1.2 ^
    bert-score==0.3.13 ^
    nltk==3.8.1 ^
    scikit-learn==1.4.2 ^
    evaluate==0.4.1 ^
    seqeval==1.2.2

echo.
echo ============================================================
echo  Installation Utilitaires
echo ============================================================
pip install ^
    pandas==2.2.2 ^
    numpy==1.26.4 ^
    matplotlib==3.8.4 ^
    seaborn==0.13.2 ^
    tqdm==4.66.2 ^
    pyyaml==6.0.1 ^
    python-dotenv==1.0.1 ^
    jupyter==1.0.0 ^
    ipywidgets==8.1.2 ^
    rich==13.7.1 ^
    loguru==0.7.2 ^
    typer==0.12.3

echo.
echo ============================================================
echo  Verification NLTK data (tokenizers)
echo ============================================================
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

echo.
echo ============================================================
echo  Verification de l'installation GPU
echo ============================================================
python -c "import torch; print('CUDA disponible :', torch.cuda.is_available()); print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun'); print('VRAM :', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1) if torch.cuda.is_available() else 'N/A', 'GB')"

echo.
echo ============================================================
echo  Verification bitsandbytes (QLoRA 4-bit)
echo ============================================================
python -c "import bitsandbytes as bnb; print('bitsandbytes OK :', bnb.__version__)"

echo.
echo ============================================================
echo  Verification ChromaDB + sentence-transformers
echo ============================================================
python -c "import chromadb; import sentence_transformers; print('ChromaDB OK'); print('SentenceTransformers OK')"

echo.
echo ============================================================
echo  SETUP TERMINE !
echo.
echo  Activer l'env   : conda activate %ENV_NAME%
echo  Lancer Jupyter  : jupyter notebook
echo  Structure du projet :
echo    RAG/
echo    |- 3GPP-clean/          (docs Rel-8 a Rel-19)
echo    |- Q_A_ID_training.csv
echo    |- questions_new.txt
echo    |- TeleQnA_testing1.txt
echo    |- TeleQnA_training.txt
echo    |- 01_rag_baseline.py
echo    |- 02_rag_optimized.py
echo    |- 03_finetune_qlora.py
echo    |- 04_ft_plus_rag.py
echo    |- 05_evaluate_all.py
echo    |- config.yaml
echo ============================================================
pause
