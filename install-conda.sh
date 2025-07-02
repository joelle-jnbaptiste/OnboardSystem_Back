#!/bin/bash

set -e  # Arrêter le script si une commande échoue

# Variables
INSTALL_DIR="$HOME/miniconda3"
ENV_NAME="py311"
PYTHON_VERSION="3.11"

echo "Téléchargement et installation de Miniconda..."

# Créer le dossier si nécessaire
mkdir -p "$INSTALL_DIR"

# Télécharger Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INSTALL_DIR/miniconda.sh"

# Installer Miniconda en mode silencieux
bash "$INSTALL_DIR/miniconda.sh" -b -u -p "$INSTALL_DIR"

# Supprimer le script d'installation
rm -f "$INSTALL_DIR/miniconda.sh"

# Initialiser conda pour bash
"$INSTALL_DIR/bin/conda" init bash

# Recharger le shell pour appliquer conda init
source ~/.bashrc

# Créer l’environnement avec Python 3.11
echo "Création de l'environnement conda '$ENV_NAME' avec Python $PYTHON_VERSION..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Activer l’environnement
echo "Activation de l'environnement '$ENV_NAME'"
conda activate "$ENV_NAME"

# Vérification
echo "Python installé : $(python --version)"
