# ============================================================
# 📦 IMPORTS
# ============================================================
import os
import multiprocessing as mp
from pathlib import Path
import torch
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import datasets  # Hugging Face Datasets

# ============================================================
# ⚙️ FIX : compatibilité Windows + Python 3.12
# ============================================================
mp.set_start_method("spawn", force=True)
os.environ["PYTORCH_NO_MEM_TRACKING"] = "1"  # évite un bug d’allocation mémoire

# ============================================================
# 🔧 CONFIGURATION GÉNÉRALE
# ============================================================
MODEL_NAME_EN_FR = "Helsinki-NLP/opus-mt-en-fr"  # modèle pré-entraîné EN→FR
MODEL_NAME_FR_EN = "Helsinki-NLP/opus-mt-fr-en"  # modèle pré-entraîné FR→EN

# Chemins des données
DATA_EN = Path("data/processed/europarl_tok.en")
DATA_FR = Path("data/processed/europarl_tok.fr")

# Répertoire de sortie pour sauvegarder les modèles fine-tunés
OUTPUT_DIR = Path("models/marianmt")

# Hyperparamètres
MAX_SAMPLES = 500     # Nombre d’exemples à charger (pour test rapide)
MAX_LENGTH = 128      # Longueur maximale des phrases (troncature)
BATCH_SIZE = 8
EPOCHS = 2
TEST_SIZE = 0.1       # 10 % des données pour la validation


# ============================================================
# 📚 1️⃣ CHARGEMENT DES DONNÉES
# ============================================================
def load_parallel_data(en_path, fr_path, max_samples=None):
    """
    Charge un corpus parallèle (anglais ↔ français) ligne par ligne.
    Chaque ligne du fichier .en correspond à la même ligne dans .fr.
    """
    if not en_path.exists() or not fr_path.exists():
        raise FileNotFoundError("Les fichiers de données anglais/français sont introuvables.")

    # Lecture limitée à `max_samples` lignes
    with en_path.open(encoding="utf-8") as f_en, fr_path.open(encoding="utf-8") as f_fr:
        en_sentences = [line.strip() for _, line in zip(range(max_samples), f_en)]
        fr_sentences = [line.strip() for _, line in zip(range(max_samples), f_fr)]

    print(f"[INFO] {len(en_sentences)} paires de phrases chargées.")
    return en_sentences, fr_sentences


# ============================================================
# 🧠 2️⃣ PRÉTRAITEMENT POUR LE TRAINER
# ============================================================
def preprocess_batch(batch, tokenizer, src_key, tgt_key, max_length=128):
    """
    Tokenise les phrases sources et cibles.
    Crée des paires (input_ids, labels) compatibles avec Seq2SeqTrainer.
    """
    # Tokenisation des phrases source et cible
    inputs = tokenizer(
        [x[src_key] for x in batch["translation"]],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    targets = tokenizer(
        [x[tgt_key] for x in batch["translation"]],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    # Le Trainer de HuggingFace s’attend à un champ "labels"
    inputs["labels"] = targets["input_ids"]
    return inputs


# ============================================================
# 🏋️‍♂️ 3️⃣ ENTRAÎNEMENT DU MODÈLE
# ============================================================
def train_model(src_lang, tgt_lang, model_name, output_subdir):
    """
    Entraîne un modèle MarianMT (EN↔FR ou FR↔EN) sur un petit corpus parallèle.
    Sauvegarde le modèle fine-tuné et le tokenizer.
    """
    print(f"\n[🚀] Entraînement du modèle {src_lang} → {tgt_lang}...")

    # --- Chargement du corpus parallèle
    en_sentences, fr_sentences = load_parallel_data(DATA_EN, DATA_FR, MAX_SAMPLES)

    # --- Sélection de la direction (EN→FR ou FR→EN)
    if src_lang == "fr" and tgt_lang == "en":
        src_sentences, tgt_sentences = fr_sentences, en_sentences
    else:
        src_sentences, tgt_sentences = en_sentences, fr_sentences

    # --- Construction du dataset HuggingFace
    dataset = datasets.Dataset.from_dict({
        "translation": [{"src": s, "tgt": t} for s, t in zip(src_sentences, tgt_sentences)]
    })

    # --- Découpage train / validation
    split = dataset.train_test_split(test_size=TEST_SIZE)
    train_dataset = split["train"]
    val_dataset = split["test"]

    # --- Chargement du tokenizer et du modèle pré-entraîné
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # --- Prétraitement des batches (tokenisation + labels)
    train_dataset = train_dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, "src", "tgt", MAX_LENGTH),
        batched=True,
        remove_columns=["translation"]
    )
    val_dataset = val_dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, "src", "tgt", MAX_LENGTH),
        batched=True,
        remove_columns=["translation"]
    )

    # --- Dossier de sortie
    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration de l’entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=20,
        save_steps=200,
        predict_with_generate=True,  # Génération automatique pour l’éval
        fp16=torch.cuda.is_available(),  # FP16 si GPU dispo
        report_to="none",  # Pas de WandB ou TensorBoard
        dataloader_num_workers=0,  # Important sous Windows
    )

    # --- Entraîneur
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # --- Lancement de l'entraînement
    trainer.train()

    # --- Sauvegarde du modèle et du tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[✅] Modèle {src_lang} → {tgt_lang} sauvegardé dans : {output_dir}")


# ============================================================
# 🧩 4️⃣ MAIN : PIPELINE COMPLET
# ============================================================
if __name__ == "__main__":
    # Entraînement EN → FR
    train_model("en", "fr", MODEL_NAME_EN_FR, "en-fr")

    # Entraînement FR → EN
    train_model("fr", "en", MODEL_NAME_FR_EN, "fr-en")

    print("\n🎯 Fine-tuning terminé pour les deux directions de traduction.")
