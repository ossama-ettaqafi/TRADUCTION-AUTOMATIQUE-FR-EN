from pathlib import Path

# Dossier des fichiers bruts
RAW_DIR = Path("data/raw")

def corpus_stats_raw(file_path: Path):
    """Affiche des statistiques sur le corpus brut"""
    if not file_path.exists():
        print(f"[⚠️] Fichier introuvable : {file_path}")
        return

    num_lines = 0
    num_words = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            if words:
                num_lines += 1
                num_words += len(words)

    print(f"[📄] Fichier brut : {file_path.name}")
    print(f"   Nombre de phrases : {num_lines}")
    print(f"   Nombre total de mots : {num_words}")
    print(f"   Longueur moyenne d'une phrase : {num_words / num_lines:.2f} mots\n")

if __name__ == "__main__":
    # Parcours tous les fichiers .en et .fr du corpus brut
    for lang in ["en", "fr"]:
        raw_file = RAW_DIR / f"europarl-v7.fr-en.{lang}"
        corpus_stats_raw(raw_file)
