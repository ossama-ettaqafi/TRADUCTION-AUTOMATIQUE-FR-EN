# ============================================================
# 📦 IMPORTS
# ============================================================
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import evaluate
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
MODEL_DIR = Path("models/marianmt/en-fr")   # Modèle MarianMT fine-tuné
DATA_SRC = Path("data/processed/europarl_tok.en")  # Corpus source
DATA_REF = Path("data/processed/europarl_tok.fr")  # Traductions de référence
MAX_SAMPLES = 200                            # Nombre de phrases à évaluer
MAX_LENGTH = 128
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement de la métrique BLEU
bleu_metric = evaluate.load("bleu")

# ============================================================
# 🧠 FONCTIONS PRINCIPALES
# ============================================================

def load_data(max_samples):
    """Charge les phrases source et de référence (alignées ligne à ligne)."""
    with DATA_SRC.open(encoding="utf-8") as f_src, DATA_REF.open(encoding="utf-8") as f_ref:
        src_sentences = [line.strip() for _, line in zip(range(max_samples), f_src)]
        ref_sentences = [line.strip() for _, line in zip(range(max_samples), f_ref)]
    return src_sentences, ref_sentences


def translate_and_analyze(model_dir, src_sentences, ref_sentences):
    """
    Traduit un échantillon de phrases avec MarianMT,
    calcule le score BLEU global et analyse les erreurs.
    """
    print(f"\n[INFO] Chargement du modèle depuis {model_dir} ({DEVICE})...")
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    translations = []

    print(f"[INFO] Génération de {len(src_sentences)} traductions...")
    for i in tqdm(range(0, len(src_sentences), BATCH_SIZE), desc="Traduction en cours"):
        batch = src_sentences[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=MAX_LENGTH)

        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translations.extend(decoded)

    # ===== ÉVALUATION BLEU =====
    global_references = [[r] for r in ref_sentences]
    results = bleu_metric.compute(predictions=translations, references=global_references)

    # ===== ANALYSE D’ERREURS =====
    error_metrics = []
    for i, (pred, ref) in enumerate(zip(translations, ref_sentences)):
        # Calcul de la précision unigramme (BLEU-1)
        result = bleu_metric.compute(predictions=[pred], references=[[ref]], max_order=1)
        unigram_precision = result['precisions'][0]

        error_metrics.append({
            'unigram_precision': unigram_precision,
            'pred': pred,
            'ref': ref,
            'src': src_sentences[i]
        })

    # Tri : du plus faible score unigramme (erreurs les plus graves)
    sorted_errors = sorted(error_metrics, key=lambda x: x['unigram_precision'])
    return results, translations, sorted_errors


# ============================================================
# 🧾 RAPPORT FINAL
# ============================================================

def main():
    print("\n🌍 ÉVALUATION DU MODÈLE MARIANMT (EN→FR)")
    print("="*60)

    # 1️⃣ Charger les données
    src_sentences, ref_sentences = load_data(MAX_SAMPLES)

    # 2️⃣ Traduire et évaluer
    results, translations, sorted_errors = translate_and_analyze(
        MODEL_DIR, src_sentences, ref_sentences
    )

    # 3️⃣ Afficher le rapport global
    print("\n===== RAPPORT D'ÉVALUATION =====")
    print(f"Nombre de phrases évaluées : {len(translations)}")
    print(f"Score BLEU global : {results['bleu'] * 100:.2f}")
    print(f"Précisions n-grammes (P1, P2, P3, P4) : {[round(p,3) for p in results['precisions']]}")
    print(f"Pénalité de brièveté : {results['brevity_penalty']:.4f}")

    # 4️⃣ Exemples d'erreurs fréquentes
    print("\n===== TOP 5 DES PIRES TRADUCTIONS =====")
    for i, err in enumerate(sorted_errors[:5]):
        print(f"\n--- Erreur #{i+1} (Précision unigramme : {err['unigram_precision']:.3f}) ---")
        print(f"[EN]  {err['src']}")
        print(f"[FR REF]  {err['ref']}")
        print(f"[FR GEN]  {err['pred']}")

    print("\n[FIN] Évaluation terminée avec succès.")
    print("="*60)


# ============================================================
# 🚀 LANCEMENT DU SCRIPT
# ============================================================
if __name__ == "__main__":
    main()
