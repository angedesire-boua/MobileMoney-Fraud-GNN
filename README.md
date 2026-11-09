# Détection de fraude Mobile Money par apprentissage sur graphes dynamiques

Projet de fin d'études en Master 2 Big Data et Intelligence Artificielle. L'étude compare des modèles tabulaires et des réseaux de neurones sur graphes pour détecter les transactions frauduleuses du jeu de données synthétique PaySim.

## Résultats principaux

| Modèle | PR-AUC sur Test |
|---|---:|
| GraphSAGE | 0,3528 |
| GNN dynamique | 0,6895 |
| Contrôle sans mémoire | 0,7499 |
| LightGBM Core | 0,9797 |

Le découpage est chronologique (entraînement 1–334, validation 335–392, test 393–743). Le choix des modèles et des seuils se fait sur la validation. Sur PaySim, le GNN dynamique améliore GraphSAGE, tandis que le contrôle sans mémoire et LightGBM Core obtiennent des résultats supérieurs.

## Contenu du dépôt

- [notebooks/](notebooks/) regroupe l'analyse de PaySim, la préparation des variables, la construction du graphe, l'entraînement et les évaluations.
- [app/](app/) contient le démonstrateur Streamlit, qui utilise LightGBM Core pour consulter et expliquer les alertes du jeu Test.
- [reports/](reports/) et [results/](results/) contiennent les figures, les analyses et les résultats tabulaires.
- [tests/](tests/) contient les tests de la visualisation du réseau des alertes.

## Lancement local

1. Avec Python 3.13, installer les dépendances de l'application via `python -m pip install -r requirements.txt`. Pour exécuter les notebooks, installer aussi les bibliothèques scientifiques indiquées dans leurs imports, notamment Jupyter, scikit-learn et PyTorch.
2. Récupérer PaySim, puis exécuter les notebooks dans l'ordre pour reconstruire les données préparées, le modèle et les artefacts de l'application. Adapter les chemins de données aux dossiers locaux si nécessaire.
3. Copier `.streamlit/secrets.example.toml` vers `.streamlit/secrets.toml` et renseigner un identifiant ainsi que l'empreinte SHA-256 d'un mot de passe de démonstration.
4. À la racine du projet, lancer `python -m streamlit run app/app.py`.

Le dataset, les modèles entraînés, les artefacts générés et la configuration de connexion locale sont conservés hors du dépôt. Le démonstrateur nécessite les fichiers produits par les notebooks.

Les tests du réseau d'alertes se lancent via `python -m unittest discover -s tests`.
