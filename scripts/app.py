# ============================================================
# 📦 IMPORTS
# ============================================================
import torch
from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path

# ============================================================
# 🔧 CONFIGURATION DES CHEMINS DES MODÈLES
# ============================================================
MODEL_DIR_EN_FR = Path("./models/marianmt/en-fr")
MODEL_DIR_FR_EN = Path("./models/marianmt/fr-en")
MAX_LENGTH = 128  # longueur maximale pour la génération

# ============================================================
# 🧠 FONCTIONS UTILES
# ============================================================

def load_model(model_dir: Path):
    """
    Charge un modèle MarianMT et son tokenizer depuis un dossier local.
    Déplace le modèle sur GPU si disponible.
    """
    model_path = str(model_dir.resolve())
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model.to("cuda")
        print(f"[INFO] Modèle chargé sur GPU depuis {model_dir}")
    else:
        print(f"[INFO] Modèle chargé sur CPU depuis {model_dir}")
        
    model.eval()  # mode évaluation
    return tokenizer, model

def translate(text: str, tokenizer, model) -> str:
    """
    Traduit un texte en utilisant le tokenizer et le modèle fournis.
    Retourne la traduction décodée.
    """
    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Génération
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=MAX_LENGTH)
    
    # Décodage
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ============================================================
# 🖥️ APPLICATION CONSOLE INTERACTIVE
# ============================================================
def main():
    print("[INFO] Chargement des modèles fine-tunés...")
    tokenizer_en_fr, model_en_fr = load_model(MODEL_DIR_EN_FR)
    tokenizer_fr_en, model_fr_en = load_model(MODEL_DIR_FR_EN)
    print("[INFO] Modèles prêts. Tapez 'quit' pour quitter.\n")

    while True:
        # Choix de la direction
        direction = input("Choisir la direction ('en->fr' ou 'fr->en') : ").strip().lower()
        if direction in ("quit", "q", "exit"):
            print("[INFO] Fermeture de l'application.")
            break
        if direction not in ("en->fr", "fr->en"):
            print("[⚠️] Direction invalide. Réessayez.\n")
            continue

        # Saisie du texte
        text = input("Texte à traduire : ").strip()
        if text.lower() in ("quit", "q", "exit"):
            print("[INFO] Fermeture de l'application.")
            break
        if not text:
            print("[⚠️] Entrée vide. Réessayez.\n")
            continue

        # Traduction
        if direction == "en->fr":
            result = translate(text, tokenizer_en_fr, model_en_fr)
        else:
            result = translate(text, tokenizer_fr_en, model_fr_en)

        # Affichage
        print("\nTraduction →", result)
        print("-" * 50)

# ============================================================
# 🔑 POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    main()
