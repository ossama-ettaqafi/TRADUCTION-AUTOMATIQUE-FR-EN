# ============================================================
# 🌍 PIPELINE COMPLET DE NETTOYAGE, ALIGNEMENT ET TOKENISATION
# ============================================================

import os
import unicodedata
import re
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 🔧 CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw")               # Corpus brut
PROCESSED_DIR = Path("data/processed")   # Corpus nettoyé + tokenisé
EMBEDDINGS_DIR = Path("data/embeddings") # Corpus encodé (IDs numériques)
VOCAB_SIZE = 16000                        # Taille du vocabulaire SentencePiece
MAX_SENTENCE_LENGTH = 128                 # Nombre max de mots par phrase

# Création des dossiers si nécessaire
for d in [PROCESSED_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🧹 FONCTIONS DE NETTOYAGE ET ALIGNEMENT
# ============================================================

def clean_sentence(sentence: str):
    """Nettoie une ligne de texte."""
    sentence = unicodedata.normalize("NFKC", sentence)
    sentence = sentence.lower()
    sentence = re.sub(r"<.*?>", " ", sentence)
    sentence = re.sub(r"[^a-zA-Z0-9\s.,;:?!'-]", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence.split()) < 2 or len(sentence.split()) > MAX_SENTENCE_LENGTH:
        return None
    return sentence


def clean_and_align_files(src_path: Path, tgt_path: Path, out_src_path: Path, out_tgt_path: Path):
    """
    Nettoie et aligne deux fichiers parallèles (source et cible)
    """
    if not src_path.exists() or not tgt_path.exists():
        print(f"[⚠️] Fichiers introuvables : {src_path}, {tgt_path}")
        return

    print(f"[🔧] Nettoyage et alignement : {src_path.name} ↔ {tgt_path.name}")

    kept = 0
    total = 0
    with src_path.open("r", encoding="utf-8") as f_src, \
         tgt_path.open("r", encoding="utf-8") as f_tgt, \
         out_src_path.open("w", encoding="utf-8") as out_src, \
         out_tgt_path.open("w", encoding="utf-8") as out_tgt:

        for src_line, tgt_line in zip(f_src, f_tgt):
            total += 1
            src_clean = clean_sentence(src_line)
            tgt_clean = clean_sentence(tgt_line)

            # Vérification de cohérence : différence de longueur < 20 mots
            if not src_clean or not tgt_clean or abs(len(src_clean.split()) - len(tgt_clean.split())) > 20:
                continue

            out_src.write(src_clean + "\n")
            out_tgt.write(tgt_clean + "\n")
            kept += 1

    print(f"[✅] Nettoyage + alignement terminé : {kept}/{total} paires conservées (~{kept/total*100:.2f}%)")


# ============================================================
# 🔤 TOKENIZER SENTENCEPIECE
# ============================================================

def train_sentencepiece(corpus_file: Path, model_prefix: str, vocab_size: int = VOCAB_SIZE):
    """
    Entraîne un modèle SentencePiece sur le corpus nettoyé.
    """
    print(f"[🚀] Entraînement du tokenizer SentencePiece pour : {corpus_file.name}")
    spm.SentencePieceTrainer.Train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram"
    )
    print(f"[✅] Tokenizer entraîné : {model_prefix}.model")


def tokenize_file(model_file: Path, input_path: Path, output_path: Path):
    """Tokenise un texte ligne par ligne avec SentencePiece."""
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))
    print(f"[✂️] Début tokenisation de {input_path.name} ({total_lines} lignes)")

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in tqdm(infile, total=total_lines, desc=f"Tokenisation {input_path.name}"):
            line = line.strip()
            if not line:
                continue
            tokens = sp.encode_as_pieces(line)
            outfile.write(" ".join(tokens) + "\n")

    print(f"[✅] Fichier tokenisé : {output_path}")


# ============================================================
# 🔢 ENCODAGE EN IDs
# ============================================================

def encode_file(model_file: Path, input_path: Path, output_path: Path):
    """Encode un fichier tokenisé en IDs numériques."""
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))
    print(f"[🔢] Encodage en IDs de {input_path.name} ({total_lines} lignes)")

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in tqdm(infile, total=total_lines, desc=f"Encodage {input_path.name}"):
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            outfile.write(" ".join(map(str, ids)) + "\n")

    print(f"[✅] Encodage terminé : {output_path}")


# ============================================================
# 🧠 PIPELINE COMPLET
# ============================================================

if __name__ == "__main__":
    print("\n==============================")
    print("🌍 DÉBUT DU PIPELINE BILINGUE")
    print("==============================")

    # Fichiers source et cible
    src_raw = RAW_DIR / "europarl-v7.fr-en.en"
    tgt_raw = RAW_DIR / "europarl-v7.fr-en.fr"
    src_clean = PROCESSED_DIR / "europarl_clean.en"
    tgt_clean = PROCESSED_DIR / "europarl_clean.fr"

    # 1️⃣ Nettoyage et alignement
    clean_and_align_files(src_raw, tgt_raw, src_clean, tgt_clean)

    # 2️⃣ Entraînement du tokenizer SentencePiece
    sp_prefix_en = PROCESSED_DIR / "spm_en"
    sp_prefix_fr = PROCESSED_DIR / "spm_fr"
    train_sentencepiece(src_clean, str(sp_prefix_en))
    train_sentencepiece(tgt_clean, str(sp_prefix_fr))

    # 3️⃣ Tokenisation
    token_file_en = PROCESSED_DIR / "europarl_tok.en"
    token_file_fr = PROCESSED_DIR / "europarl_tok.fr"
    tokenize_file(sp_prefix_en.with_suffix(".model"), src_clean, token_file_en)
    tokenize_file(sp_prefix_fr.with_suffix(".model"), tgt_clean, token_file_fr)

    # 4️⃣ Encodage en IDs
    emb_file_en = EMBEDDINGS_DIR / "europarl_emb.en"
    emb_file_fr = EMBEDDINGS_DIR / "europarl_emb.fr"
    encode_file(sp_prefix_en.with_suffix(".model"), token_file_en, emb_file_en)
    encode_file(sp_prefix_fr.with_suffix(".model"), token_file_fr, emb_file_fr)

    print("\n🎉 Pipeline complet exécuté avec succès !")
