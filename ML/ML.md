# Formation Machine Learning

Bienvenue dans ce dépôt d’exercices pour la formation en Machine Learning !  
Cette formation a pour but d’explorer les concepts fondamentaux du machine learning à travers des implémentations pratiques en Python et l’utilisation de bibliothèques populaires comme `scikit-learn`, `numpy`, `pytorch`, etc.

## Contenu

Chaque répertoire aborde un thème ou un algorithme particulier du ML. Voici un aperçu des sujets couverts :

- Régression linéaire
- Régression polynomiale
- Régression logistique
- Arbres de décision, Forêts aléatoires et Gradient Boosting
- Réseaux de neurones

## Structure typique d’un exercice

Chaque exercice suit généralement la structure suivante :

1. **Séparation en jeu d’entraînement / validation**
2. **Implémentation et entraînement du modèle**
3. **Évaluation du modèle avec des métriques pertinentes**

## Notebook
Un notebook Jupyter des exercices de ce dépôt est également mis à votre disposition.
Nous vous suggérons fortement de l’utiliser avec [Google colab](https://https://colab.research.google.com) si vous n’êtes pas à l’aise avec l’exécution des scripts en ligne de commande ou si vous préférez un environnement interactif pour coder, tester et visualiser les résultats pas à pas.

## Pré-requis

Avant de commencer, assurez‑vous d’avoir :
- **Suivi le MOOC** : [Initiez‑vous au Machine Learning](https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning) (OpenClassrooms)
- **Lu les chapitres 1 à 4** de la série [Neural Networks](https://www.3blue1brown.com/topics/neural-networks) (3Blue1Brown)
- **Réalisé le tutoriel** : [PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- **Python 3.8 +** installé
- **Dépendances du projet** installées :

```bash
pip install numpy matplotlib scikit-learn torch torchvision
```

## Exécution des exercices

Chaque exercice possède un fichier que vous pouvez exécuter directement à partir de la racine du dépôt :

```bash
python src/linear_regression.py # exemple
```

Remplacez le chemin par celui de l'exercice que vous souhaitez lancer.

## Lancement des tests

Les tests unitaires sont écrits avec **unittest**. Depuis la racine du dépôt :

```bash
cd ML/tests
PYTHONPATH=../src python -m unittest discover
```

Vous pouvez également exécuter un seul fichier de tests :

```bash
cd ML/tests
PYTHONPATH=../src python -m unittest test_linear_regression.py # exemple
```

## Pour aller plus loin

Voici quelques ressources pour approfondir vos connaissances en Machine Learning au-delà de ce qui a été présenté dans ce dépôt:

- [MIT 6.036 - Introduction to Machine Learning](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/) ⭐⭐
- [MIT 6.S191 - Introduction to Deep Learning](https://www.youtube.com/watch?v=alfdI7S6wCY&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) ⭐⭐
- [MIT 18.657 - Mathematics of Machine Learning](https://ocw.mit.edu/courses/18-657-mathematics-of-machine-learning-fall-2015/download/) ⭐⭐⭐
- [PolyMTL INF8245E - Machine Learning (Vidéos)](https://www.youtube.com/watch?v=-6ChHxllZVU&list=PLImtCgowF_ETupFCGQqmvS_2nqErZbifm) ⭐⭐
- [PolyMTL INF8245E - Machine Learning (Notes)](https://drive.google.com/drive/folders/1xUqzxJK5NbAxUZOInpTBAAQk8iwKSKVS) ⭐⭐
- [PolyMTL INF8359DE - Reinforcement Learning](https://www.youtube.com/watch?v=J9JZyyPCJcQ&list=PLImtCgowF_ES_JdF_UcM60EXTcGZg67Ua) ⭐⭐
- [PolyMTL MTH3302 - Méthodes probabilistes et statistiques pour l'IA](https://github.com/decorJim/mth3302) ⭐⭐⭐
- [An Introduction to Statistical Learning — Gareth James, Daniela Witten, Trevor Hastie & Robert Tibshirani](https://www.statlearning.com/) ⭐⭐
- [Statistical Learning Theory — Vladimir N. Vapnik](https://www.wiley.com/en-us/Statistical-Learning-Theory-9780471030034) ⭐⭐⭐

Légende:
- ⭐: pas (ou peu) de pré‑requis en ML/programmation/maths.
- ⭐⭐: notions de base acquises; à l’aise avec Python + algèbre linéaire + probabilités.
- ⭐⭐⭐: solide bagage mathématique (analyse, probabilités, statistiques, optimisation) et envie de rigueur théorique.