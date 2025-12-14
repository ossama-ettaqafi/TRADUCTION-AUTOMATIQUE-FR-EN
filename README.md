# 🌍 Traduction Automatique (Français-Anglais)

Un système complet de traduction automatique neuronale utilisant des modèles MarianMT fine-tunés sur le corpus Europarl.

## 📋 Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation Rapide](#-utilisation-rapide)
- [Guide Détaillé](#-guide-détaillé)
- [Résultats](#-résultats)
- [API](#-api)
- [Dépannage](#-dépannage)

## 🎯 Aperçu du Projet

Ce projet implémente un pipeline complet pour la traduction automatique bidirectionnelle français-anglais. Le système utilise l'architecture MarianMT de Hugging Face, fine-tunée sur le corpus Europarl pour des performances optimisées.

### Technologies Utilisées

- **🤗 Transformers** : Modèles MarianMT pré-entraînés
- **⚡ PyTorch** : Backend d'entraînement et d'inférence
- **🔤 SentencePiece** : Tokenisation sous-mots
- **📊 Europarl** : Corpus parallèle de qualité
- **🌐 Flask** : Interface web

## ✨ Fonctionnalités

### 🔄 Traduction Bidirectionnelle
- **Français → Anglais**
- **Anglais → Français**
- Support d'autres paires de langues (extensible)

### 🛠️ Pipeline Complet
- **Nettoyage** et alignement des données
- **Tokenisation** avancée avec SentencePiece
- **Entraînement** avec fine-tuning
- **Évaluation** automatique avec métrique BLEU
- **Déploiement** via interface console et web

### 📈 Évaluation Détaillée
- Score BLEU et précisions n-grammes
- Analyse qualitative des erreurs
- Comparaison cible/prédiction
- Métriques par phrase

## 🏗️ Architecture

```
traduction-automatique/
├── 📁 data/
│   ├── 📁 raw/                 # Données brutes Europarl
│   ├── 📁 processed/           # Données nettoyées
│   └── 📁 embeddings/          # Encodages numériques
├── 📁 models/
│   └── 📁 marianmt/
│       ├── 📁 en-fr/          # Modèle Anglais→Français
│       └── 📁 fr-en/          # Modèle Français→Anglais
├── 📁 web/                    # Application Flask
├── 🔧 explore.py              # Analyse des données
├── 🧹 preprocessing.py        # Prétraitement
├── 🏋️‍♂️ train.py               # Entraînement
├── 📊 evaluate_bleu.py        # Évaluation
├── 💻 app.py                  # Interface console
└── 📖 README.md
```

## ⚙️ Installation

### Prérequis Système

- Python 3.8+
- 8GB+ RAM (16GB recommandé)
- GPU NVIDIA (optionnel mais recommandé pour l'entraînement)

### Installation des Dépendances

```bash
# Cloner le repository
git clone <votre-repo>
cd traduction-automatique

# Créer un environnement virtuel (optionnel)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install torch transformers datasets sentencepiece tqdm evaluate flask

# Vérifier l'installation
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### Téléchargement des Données

1. **Télécharger Europarl v7** :
```bash
# Télécharger depuis https://www.statmt.org/europarl/
mkdir -p data/raw
# Placer les fichiers dans data/raw/
# - europarl-v7.fr-en.en
# - europarl-v7.fr-en.fr
```

## 🚀 Utilisation Rapide

### Mode Express (Modèles Pré-entraînés)

```bash
# Utiliser directement les modèles Hugging Face
python app.py
```

### Pipeline Complet (Recommandé)

```bash
# 1. Analyse des données
python explore.py

# 2. Prétraitement (30-60 minutes)
python preprocessing.py

# 3. Entraînement (2-4 heures avec GPU)
python train.py

# 4. Évaluation
python evaluate_bleu.py

# 5. Utilisation
python app.py
```

## 📚 Guide Détaillé

### 1. Exploration des Données

```bash
python explore.py
```
**Sortie attendue :**
```
[📄] Fichier brut : europarl-v7.fr-en.en
   Nombre de phrases : 2,000,000
   Nombre total de mots : 45,000,000
   Longueur moyenne : 22.5 mots
```

### 2. Prétraitement Avancé

Le script `preprocessing.py` effectue :

- **Nettoyage** : Normalisation, suppression HTML, filtrage
- **Alignement** : Paires parallèles cohérentes
- **Tokenisation** : SentencePiece avec vocabulaire 16k
- **Encodage** : Conversion en IDs numériques

### 3. Entraînement des Modèles

**Configuration par défaut :**
```python
BATCH_SIZE = 8
EPOCHS = 2
MAX_LENGTH = 128
MAX_SAMPLES = 500  # Ajuster selon vos ressources
```

**Pour un entraînement complet :**
```python
MAX_SAMPLES = 500000  # 500k échantillons
EPOCHS = 3
BATCH_SIZE = 16  # Si GPU avec >8GB VRAM
```

### 4. Évaluation des Performances

```bash
python evaluate_bleu.py
```

**Métriques fournies :**
- Score BLEU (1-4 grammes)
- Précisions individuelles
- Pénalité de brièveté
- Top 5 des erreurs

## 📊 Résultats

### Performances Typiques

| Modèle | BLEU Score | Précision 1-gram | Temps d'Entraînement |
|--------|------------|------------------|---------------------|
| Base (Helsinki-NLP) | ~35.2 | ~0.55 | - |
| Fine-tuné (500 éch.) | ~32.4 | ~0.52 | 30 min |
| Fine-tuné (50k éch.) | ~38.1 | ~0.61 | 4 heures |

### Exemple de Traduction

**Entrée (EN) :** "The committee will examine the proposal next week."
**Sortie (FR) :** "La commission examinera la proposition la semaine prochaine."

## 🌐 API et Interfaces

### Interface Console

```bash
python app.py

# Sortie :
Choisir la direction ('en->fr' ou 'fr->en') : en->fr
Texte à traduire : Hello, how are you?
Traduction → Bonjour, comment allez-vous ?
```

### Application Web Flask

```bash
cd web
python app.py
# Accéder à http://localhost:5000
```

### API REST

```python
import requests

response = requests.post(
    "http://localhost:5000/translate",
    json={
        "text": "This is a test sentence.",
        "direction": "en->fr"
    }
)
print(response.json()["translation"])
```

## 🔧 Configuration Avancée

### Hyperparamètres d'Entraînement

Modifier dans `train.py` :

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=4,           # Plus d'époques
    per_device_train_batch_size=16, # Batch plus grand
    learning_rate=5e-5,           # Taux d'apprentissage
    warmup_steps=500,             # Warmup
    weight_decay=0.01,            # Régularisation
    fp16=True,                    # Acceleration GPU
)
```

### Tokenisation Personnalisée

Dans `preprocessing.py` :

```python
# Pour d'autres langues
VOCAB_SIZE = 32000
MODEL_TYPE = "bpe"  # "unigram" ou "bpe"
CHARACTER_COVERAGE = 1.0  # Pour couvrir tous les caractères
```

## 🐛 Dépannage

### Problèmes Courants

**Erreur Mémoire Insuffisante**
```python
# Solution : Réduire la configuration
BATCH_SIZE = 4
MAX_LENGTH = 64
MAX_SAMPLES = 100
```

**Fichiers Manquants**
```bash
# Vérifier la structure
ls data/raw/
# Devrait afficher : europarl-v7.fr-en.en et europarl-v7.fr-en.fr
```

**Entraînement Trop Lent**
- Activer CUDA : `torch.cuda.is_available()`
- Utiliser FP16 : `fp16=True`
- Réduire `MAX_SAMPLES`

### Extensions Possibles

1. **Nouvelles Paires de Langues**
```python
# Ajouter dans train.py
MODEL_CONFIGS = {
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    # ...
}
```

2. **Interface Graphique**
- Streamlit pour le prototypage
- Gradio pour démonstrations
- Interface React avancée

3. **Déploiement Production**
- Container Docker
- API FastAPI
- Scaling avec multiples GPUs

## 🤝 Contribution

Les contributions sont bienvenues ! Voici comment participer :

1. **Signaler un bug** : Ouvrir une issue avec les étapes pour reproduire
2. **Suggérer une amélioration** : Proposer de nouvelles fonctionnalités
3. **Soumettre du code** : Pull request avec tests et documentation

### Développement

```bash
# Setup développement
git clone <repo>
cd traduction-automatique
pip install -e .[dev]

# Lancer les tests
python -m pytest tests/

# Vérifier le style de code
flake8 scripts/
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

**Note importante** : Les modèles pré-entraînés Helsinki-NLP sont sous leur propre licence. Consultez les conditions d'utilisation sur [Hugging Face](https://huggingface.co/Helsinki-NLP).

## 🙏 Remerciements

- **Hugging Face** pour l'excellente bibliothèque Transformers
- **Union Européenne** pour le corpus Europarl
- **Communauté Open Source** pour les outils et ressources

*✨ Fait avec passion pour l'apprentissage automatique et la linguistique computationnelle ✨*
