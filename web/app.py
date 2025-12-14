# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Chemins vers les modèles fine-tunés
MODEL_DIR_EN_FR = Path("./models/marianmt/en-fr")
MODEL_DIR_FR_EN = Path("./models/marianmt/fr-en")

# Variables globales pour les modèles
tokenizer_en_fr = None
model_en_fr = None
tokenizer_fr_en = None
model_fr_en = None
models_loaded = False

def load_fine_tuned_models():
    """Charger les modèles fine-tunés depuis les dossiers locaux"""
    global tokenizer_en_fr, model_en_fr, tokenizer_fr_en, model_fr_en, models_loaded
    
    logger.info("Chargement des modèles fine-tunés...")
    
    try:
        # Vérifier que les dossiers existent
        if not MODEL_DIR_EN_FR.exists():
            raise FileNotFoundError(f"Dossier modèle EN->FR introuvable: {MODEL_DIR_EN_FR}")
        if not MODEL_DIR_FR_EN.exists():
            raise FileNotFoundError(f"Dossier modèle FR->EN introuvable: {MODEL_DIR_FR_EN}")
        
        # Charger le modèle English -> French
        logger.info(f"Chargement du modèle EN->FR depuis {MODEL_DIR_EN_FR}")
        tokenizer_en_fr = MarianTokenizer.from_pretrained(str(MODEL_DIR_EN_FR))
        model_en_fr = MarianMTModel.from_pretrained(str(MODEL_DIR_EN_FR))
        
        # Charger le modèle French -> English
        logger.info(f"Chargement du modèle FR->EN depuis {MODEL_DIR_FR_EN}")
        tokenizer_fr_en = MarianTokenizer.from_pretrained(str(MODEL_DIR_FR_EN))
        model_fr_en = MarianMTModel.from_pretrained(str(MODEL_DIR_FR_EN))
        
        # Déplacer sur GPU si disponible
        if torch.cuda.is_available():
            model_en_fr = model_en_fr.to("cuda")
            model_fr_en = model_fr_en.to("cuda")
            logger.info("Modèles déplacés sur GPU")
        else:
            logger.info("Modèles sur CPU")
            
        # Mode évaluation
        model_en_fr.eval()
        model_fr_en.eval()
        
        models_loaded = True
        logger.info("✅ Tous les modèles fine-tunés sont chargés avec succès!")
        
        # Afficher quelques informations sur les modèles
        logger.info(f"Modèle EN->FR: {model_en_fr.config}")
        logger.info(f"Modèle FR->EN: {model_fr_en.config}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
        models_loaded = False
        raise

@app.before_request
def before_first_request():
    """Charger les modèles avant la première requête"""
    global models_loaded
    
    if not models_loaded:
        try:
            load_fine_tuned_models()
        except Exception as e:
            logger.error(f"Échec du chargement des modèles: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "healthy" if models_loaded else "error",
        "models_loaded": models_loaded,
        "models_type": "fine-tuned",
        "supported_pairs": ["en->fr", "fr->en"]
    })

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    """Endpoint de traduction"""
    if not models_loaded:
        return jsonify({'error': 'Modèles de traduction non disponibles.'}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400
        
        text = data.get('text', '').strip()
        source_lang = data.get('sourceLang', '')
        target_lang = data.get('targetLang', '')
        
        # Validation des entrées
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        if len(text) > 2000:
            return jsonify({'error': 'Texte trop long. Maximum 2000 caractères.'}), 400
        
        # Validation de la paire de langues
        valid_pairs = [('en', 'fr'), ('fr', 'en')]
        if (source_lang, target_lang) not in valid_pairs:
            return jsonify({'error': 'Paire de langues non supportée'}), 400
        
        # Sélection du modèle et tokenizer approprié
        if source_lang == 'en' and target_lang == 'fr':
            tokenizer = tokenizer_en_fr
            model = model_en_fr
            direction = "EN → FR"
        else:  # fr -> en
            tokenizer = tokenizer_fr_en
            model = model_fr_en
            direction = "FR → EN"
        
        # Tokenisation
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Déplacement sur GPU si disponible
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Génération de la traduction
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
        
        # Décodage de la traduction
        translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        logger.info(f"Traduction réussie {direction}: {len(text)} caractères")
        
        return jsonify({
            'translation': translation,
            'sourceLang': source_lang,
            'targetLang': target_lang,
            'characterCount': len(text),
            'modelType': 'fine-tuned'
        })
        
    except Exception as e:
        logger.error(f"Erreur API: {e}")
        return jsonify({'error': 'Échec de la traduction. Veuillez réessayer.'}), 500

@app.route('/model-info')
def model_info():
    """Information sur les modèles"""
    if not models_loaded:
        return jsonify({'error': 'Modèles non chargés'}), 503
    
    return jsonify({
        'en_fr_model': {
            'path': str(MODEL_DIR_EN_FR),
            'vocab_size': model_en_fr.config.vocab_size,
            'model_type': model_en_fr.config.model_type
        },
        'fr_en_model': {
            'path': str(MODEL_DIR_FR_EN),
            'vocab_size': model_fr_en.config.vocab_size,
            'model_type': model_fr_en.config.model_type
        }
    })

if __name__ == '__main__':
    # Charger les modèles au démarrage
    try:
        load_fine_tuned_models()
        logger.info("🚀 Serveur Flask démarré avec les modèles fine-tunés")
    except Exception as e:
        logger.error(f"❌ Impossible de démarrer le serveur: {e}")
        exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=5000)