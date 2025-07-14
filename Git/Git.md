# Formation Git : Commandes Essentielles et Workflows

La formation suivante est un guide complet pour maîtriser le système de contrôle de version Git à travers des exemples pratiques et des explications claires.

## Table des Matières

1. [Démarrage](#démarrage)
2. [Opérations de Base](#opérations-de-base)
3. [Travailler avec les Modifications](#travailler-avec-les-modifications)
4. [Branches et Fusion](#branches-et-fusion)
5. [Opérations Distantes](#opérations-distantes)
6. [Exercices Pratiques](#exercices-pratiques)
7. [Bonnes Pratiques](#bonnes-pratiques)

## Démarrage

### Prérequis

- [Git](https://git-scm.com/) installé sur votre système
- Connaissances de base de la ligne de commande
- Un compte [GitHub](https://github.com/)

### Qu'est-ce que Git ?

Git est un système de contrôle de version distribué qui suit les modifications dans les fichiers et coordonne le travail entre plusieurs développeurs. Il vous permet de sauvegarder des instantanés de votre projet à différents moments et de collaborer efficacement avec d'autres.

### Qu'est-ce qu'un commit ?

Un **commit** est l'élément fondamental de Git. C'est un instantané de votre projet à un moment donné. Imaginez que vous prenez une photo de tous vos fichiers et leurs contenus à un instant précis: c'est exactement ce qu'est un commit.

**Caractéristiques d'un commit :**
- **Instantané complet** : Chaque commit contient l'état de tous les fichiers de votre projet
- **Identifiant unique** : Chaque commit a un hash SHA (ex: `a1b2c3d4...`) qui l'identifie de façon unique
- **Métadonnées** : Auteur, date, message descriptif
- **Lien avec le parent** : Chaque commit "connaît" le commit précédent, créant une chaîne d'historique
- **Immuable** : Une fois créé, un commit ne peut pas être modifié (seulement remplacé)

**Analogie simple :**
Pensez à un commit comme à une version sauvegardée d'un document Word avec la fonction "Suivi des modifications". Chaque fois que vous sauvegardez, vous créez un point de restauration avec un commentaire expliquant ce qui a changé.

**Exemple visuel d'historique de commits :**
```
A1b2c3 - "Ajouter fonction de connexion" (il y a 2 heures)
B4e5f6 - "Corriger bug dans la validation" (hier)
C7g8h9 - "Créer interface utilisateur" (il y a 2 jours)
D0i1j2 - "Commit initial" (il y a 3 jours)
```

Chaque commit est un point stable dans l'historique de votre projet auquel vous pouvez revenir à tout moment.

### Concepts Clés à Comprendre

#### Le Répertoire de Travail, la Zone de Staging et le Dépôt

Git fonctionne avec trois espaces principaux :

1. **Répertoire de travail** : Vos fichiers actuels sur votre ordinateur
2. **Zone de staging** : Zone intermédiaire où vous préparez votre prochain commit
3. **Dépôt local** : Où Git stocke l'historique de vos commits

**Flux de travail typique :**
```
Répertoire de travail → [Ajout de fichiers] → Zone de staging → [Création d'un commit] → Dépôt local
```

#### Qu'est-ce qu'un Dépôt (Repository) ?

Un **dépôt** est un dossier qui contient :
- Vos fichiers de projet
- L'historique complet des modifications (dans le dossier caché `.git`)
- Les informations de configuration

**Types de dépôts :**
- **Dépôt local** : Sur votre ordinateur
- **Dépôt distant** : Sur un serveur (GitHub, GitLab, etc.)

#### Qu'est-ce qu'une Branche ?

Une **branche** est une ligne de développement indépendante. Imaginez un arbre :
- Le **tronc** = branche principale (souvent appelée `main` ou `master`)
- Les **branches** = nouvelles fonctionnalités ou expérimentations

**Pourquoi utiliser des branches ?**
- Développer des fonctionnalités sans affecter le code principal
- Permettre à plusieurs développeurs de travailler simultanément
- Faciliter les tests et l'expérimentation

#### Dépôts Locaux vs Distants

**Dépôt local :**
- Sur votre machine
- Où vous faites vos modifications
- Historique complet disponible hors ligne

**Dépôt distant :**
- Sur un serveur (GitHub, GitLab, etc.)
- Point de partage avec l'équipe
- Sauvegarde de votre travail

**Synchronisation :**
- `git push` : Envoie vos commits locaux vers le distant
- `git pull` : Récupère les commits distants vers votre local

## Opérations de Base

### `git clone`

Télécharge un dépôt depuis une source distante vers votre machine locale.

1. Exécutez la commande suivante pour cloner le dépôt de cette formation:
```bash
# Exemple: Cloner le dépôt de cette formation
git clone https://github.com/HekaPoly/BIRA_onboarding.git
```

2. Naviguez dans le dossier créé en exécutant la commande suivante:
```bash
cd BIRA_onboarding
```

**Quand l'utiliser :** Pour commencer à travailler sur un projet existant ou créer une copie locale d'un dépôt distant.

### `git status`

Affiche l'état actuel de votre répertoire de travail et de la zone de staging.

1. Modifiez le fichier `membres.txt` en y ajoutant votre nom sur une nouvelle ligne. Vous pouvez exécuter la commande suivante:

```bash
echo "Votre nom" >> membres.txt
```

2. Créez un fichier texte vide. Vous pouvez exécuter la commande suivante:
```bash
touch empty.txt
```

3. Maintenant exécutez la commande suivante:
```bash
git status
```

**Explication de la sortie :**
- **Fichiers non suivis :** Nouveaux fichiers pas encore ajoutés à Git (`empty.txt`)
- **Fichiers modifiés :** Fichiers qui ont été modifiés depuis le dernier commit (`membres.txt`)
- **Fichiers stagés :** Fichiers prêts à être committés (Ici aucun puisque nous n'avons pas dit à Git de stager nos modifications)

**Rappel :** Un commit est un instantané permanent de votre projet. Avant de créer ce commit, vous devez choisir quelles modifications inclure (c'est le processus de "staging" avec `git add`).

**Quand l'utiliser :** Fréquemment ! Utilisez cette commande pour comprendre quelles modifications vous avez et ce que Git va faire ensuite.

### `git add`

Met en staging les modifications pour le prochain commit.

1. Ajoutez en staging les modifications que vous avez faites au fichier membres.txt en exécutant la commande suivante:
```bash
# Ajouter un fichier spécifique
git add membres.txt
```

2. Exécutez la commande suivante:
```bash
git status
```

Observez que vous avez maintenant des fichiers en staging prêts à être commit!

**Quand l'utiliser :** Avant de committer des modifications. Seules les modifications stagées seront incluses dans le commit.

### `git rm`

Supprime des fichiers du répertoire de travail et de la zone de staging.

1. Supprimez le fichier `empty.txt` que vous avez créé en exécutant la commande suivante:
```bash
# Supprime votre fichier
git rm empty.txt
```

2. Exécutez la commande suivante:
```bash
git status
```

Observez que vous n'avez plus de fichiers non suivis dans la sortie de la commande.

**Quand l'utiliser :** Quand vous voulez supprimer des fichiers de votre projet et faire en sorte que Git suive cette suppression.

### `git restore`

Restaure les fichiers à leur état précédent ou annule les modifications stagées.

1. Restaurer le fichier membres.txt en exécutant la commande suivante:
```bash
# Restaurer le fichier membres.txt à son dernier état committé
git restore membres.txt
```

2. Exécutez la commande suivante:
```bash
git status
```

Observez que les modifications apportés au fichier `membres.txt` ont été annulées!

**Quand l'utiliser :** Quand vous voulez annuler des modifications que vous avez apportées aux fichiers ou déstagé des fichiers que vous avez ajoutés.

## Travailler avec les Modifications

### `git commit`

Crée un instantané de vos modifications stagées.

1. Ajouter une ligne avec votre nom au fichier `membres.txt` en exécutant la commande suivante (Nous reviendrons sur la commande `git checkout` plus bas):
```bash
git checkout -b mon_premier_commit && echo "Votre nom" >> membres.txt
```

2. Ajouter vos changements à la zone de staging en exécutant la commande suivante:
```bash
git add membres.txt
```

3. Commitez vos changements au fichier `membres.txt` en exécutant la commande suivante. Remplacez le message du commit avec un message décrivant vos changements (Par exemple: Ajout de mon nom):
```bash
# Commit avec un message
git commit -m "Message du commit"
```

**Bonnes pratiques pour les messages de commit :**
- Utilisez le présent ("Ajouter fonctionnalité" et non "Ajouté fonctionnalité")
- Gardez la première ligne sous 50 caractères
- Utilisez le corps pour expliquer quoi et pourquoi, pas comment

### `git log`

Affiche l'historique des commits.

```bash
# Log basique
git log
```

Observez l'historique de vos commits.

**Quand l'utiliser :** Pour examiner l'historique du projet, trouver des commits spécifiques, ou comprendre quelles modifications ont été apportées.

## Branches et Fusion

### `git branch`

Crée, liste et gère les branches.

1. Listez les branches en exécutant la commande suivante:
```bash
# Lister toutes les branches
git branch
```

Vous devriez avoir normalement au moins 2 branches: `main` et `mon_premier_commit`.

2. Crééz votre première branche en exécutant la commande suivante:
```bash
# Créer une branche
git branch ma_premiere_branche
```

Si vous réexécutez `git branch` vous devriez voir votre branche nouvellement créée dans la liste des branches locales

**Quand l'utiliser :** Pour organiser différentes fonctionnalités, expérimentations ou versions de votre code.

### `git checkout`

Bascule entre les branches ou restaure des fichiers.

1. Exécutez la commande suivante:
```bash
git status
```

Observez que vous êtes actuellement sur la branche `mon_premier_commit`.

2. Basculez sur la branche main en exécutant la commande suivante:
```bash
# Basculer vers la branche main
git checkout main
```

3. Réexécutez la commande suivante:
```bash
git status
```

Observez que vous êtes maintenant sur la branche `main`!

En ce qui concerne la commande `git checkout -b nom_branche` utilisée plus haut, l'ajout de l'argument `-b` permet de créer une branche et de basculer automatiquement dessus!. C'est une combinaison de `git branch nom_branche` et `git checkout nom_branche`.

**Quand l'utiliser :** Pour basculer entre différentes lignes de développement ou pour restaurer des fichiers depuis différents commits.

### `git merge`

Combine les modifications de différentes branches.

1. Basculez sur la branche `ma_premiere_branche` en exécutant la commande suivante:
```bash
git checkout ma_premiere_branche
```

2. Fusionnez la branche `mon_premier_commit` à la branche `ma_premiere_branche` en exécutant la commande suivante:
```bash
# Fusionner la branche mon_premier_commit dans la branche ma_premiere_branche
git merge mon_premier_commit
```

`git merge` créera alors ce qu'on appelle un merge commit. C'est un commit contenant les changements de la branche `mon_premier_commit` qui ne sont pas sur la branche `ma_premiere_branche`. Un éditeur de texte s'ouvrira pour que vous puissiez modifer le message du commit; veuillez simplement le fermer.

3. Exécutez la commande suivante:
```bash
git log
```

Observez le nouveau commit créé sur votre branche `ma_premiere_branche`. Remarquez également que vous avez maintenant les changements de la branche `mon_premier_commit` sur votre branche!

**Quand l'utiliser :** Pour intégrer des fonctionnalités terminées dans votre branche principale.

**Gestion des conflits de fusion :**
1. Git marquera les fichiers en conflit
2. Éditez les fichiers pour résoudre les conflits
3. Stagez les fichiers résolus avec `git add`
4. Complétez la fusion avec `git commit`

## Opérations Distantes

### `git fetch`

Télécharge les modifications depuis un dépôt distant sans les fusionner.

1. Exécutez la commande suivante:
```bash
# Fetch depuis origin
git fetch
```

Observez la sortie de la commande. Si des modifications ont été faites au dépôt entre le moment où vous l'avez cloné et le moment où vous avez exécutez cette commande, vous devriez voir que de nouvelles branches distantes ont été créées.

**Quand l'utiliser :** Pour voir quelles modifications sont disponibles sur le dépôt distant sans affecter votre travail local.

### `git pull`

Fetch et fusionne les modifications depuis un dépôt distant. `git pull` est équivalent à exécuter `git fetch`, puis `git merge`.

1. Basculez sur la branche `main` en exécutant la commande suivante:
```bash
git checkout main
```

2. Exécutez la commande suivante:
```bash
# Pull depuis l'upstream de la branche courante
git pull
```

Observez la sortie de la commande. Si votre branche est `main` est déjà à jour avec la branche distante vous devriez voir le message `Already up to date`. Sinon, vous serez en mesure d'observer les opérations de `fetch` et de `merge` dans la sortie.

**Quand l'utiliser :** Pour mettre à jour votre branche locale avec les dernières modifications du dépôt distant.

### `git push`

Télécharge vos commits locaux vers un dépôt distant.

1. Créez une branche à votre nom et basculez dessus:
```bash
git checkout -b votre_nom
```

2. Ajoutez votre nom au fichier `membres.txt` en exécutant la commande suivante:
```bash
echo "Votre nom" >> membres.txt
```

3. Ajoutez vos changements à la zone de staging en exécutant la commande suivante:
```bash
git add membres.txt
```

4. Commitez vos changements:
```
git commit -m "Votre message de commit"
```

5. Poussez vos changements sur le dépôt distant:
```bash
# Push vers l'upstream de la branche courante
git push
```

Bravo! Maintenant vos changements sont suivis sur votre dépôt local ET sur le dépôt distant sur la plateforme GitHub!

**Quand l'utiliser :** Pour partager vos commits avec d'autres ou pour mettre à jour le dépôt distant.

## Exercices Pratiques

### Exercice 1 : Conflits de Fusion Simples

#### Préparation Rapide

```bash
# 1. Vérifier que vous êtes dans un dépôt Git et basculez sur la branche main
git status && git checkout main

# 2. Créer un fichier simple
echo "Bonjour monde" > salutation.txt
git add salutation.txt
git commit -m "Ajout salutation initiale"

# 3. Créer deux branches avec des modifications conflictuelles
git checkout -b branche-francais
echo "Bonjour tout le monde!" > salutation.txt
git add salutation.txt
git commit -m "Salutation en français"

git checkout main
echo "Hello world!" > salutation.txt
git add salutation.txt
git commit -m "Salutation en anglais"
```

#### Exercice de Merge avec Conflit

```bash
# 1. Essayer de fusionner - ceci créera un conflit
git merge branche-francais

# 2. Git vous avertira d'un conflit. Examiner le fichier
cat salutation.txt
# Vous verrez :
# <<<<<<< HEAD
# Hello world!
# =======
# Bonjour tout le monde!
# >>>>>>> branche-francais

# 3. Résoudre le conflit en éditant le fichier
echo "Hello world! Bonjour tout le monde!" > salutation.txt

# 4. Finaliser la fusion
git add salutation.txt
git commit -m "Fusion des salutations"
```

#### Points Clés

**Identifier un conflit :**
- Git s'arrête et affiche un message d'erreur
- `git status` montre les fichiers en conflit
- Les marqueurs `<<<<<<<`, `=======`, `>>>>>>>` délimitent les conflits

**Résoudre un conflit :**
1. Éditer le fichier pour choisir/combiner les versions
2. Supprimer les marqueurs de conflit
3. `git add` le fichier résolu
4. `git commit` (pour merge)

### Exercice 2 : Créer une Pull Request sur GitHub

Cet exercice vous guide à travers le processus complet de création d'une Pull Request (PR) sur GitHub.

#### Étape 1 : Créer la Pull Request sur GitHub

**Interface Web GitHub :**

1. **Aller sur GitHub** : Ouvrez votre navigateur et allez sur votre dépôt GitHub
2. **Notification automatique** : GitHub affichera une bannière jaune proposant "Compare & pull request"
3. **Cliquer sur "Compare & pull request"** ou aller dans l'onglet "Pull requests" → "New pull request"

**Remplir les informations de la PR :**

#### Étape 2 : Paramètres de la Pull Request

**Configurations importantes :**

1. **Base branch** : `main` (branche de destination)
2. **Compare branch** : `feature/amelioration-readme` (votre branche)
3. **Reviewers** : Ajouter des collègues ou mainteneurs
4. **Assignees** : Vous assigner ou assigner la personne responsable
5. **Labels** : Ajouter des labels appropriés (ex: `documentation`, `enhancement`)

#### Bonnes pratiques pour les Pull Requests

**Avant de créer une PR :**
- Tester votre code localement
- Vérifier que les tests passent (si applicable)
- Relire vos changements
- S'assurer que la branche est à jour avec `main`

**Titre et description :**
- Titre clair et descriptif
- Description détaillée expliquant le "pourquoi"
- Lister les changements principaux
- Inclure des captures d'écran si pertinent

**Pendant la révision :**
- Répondre aux commentaires rapidement
- Appliquer les suggestions constructives
- Expliquer vos choix de design si nécessaire

## Bonnes Pratiques

### Directives pour les Commits

1. **Faire des commits atomiques :** Chaque commit devrait représenter un changement logique unique
2. **Écrire des messages de commit clairs :** Expliquez quoi et pourquoi, pas comment
3. **Committer fréquemment :** Les petits commits fréquents sont plus faciles à comprendre et à annuler
4. **Tester avant de committer :** Assurez-vous que votre code fonctionne avant de committer

### Gestion des Branches

1. **Utiliser des noms de branches descriptifs :** `feature/authentification-utilisateur`, `bugfix/erreur-connexion`
2. **Garder les branches courtes :** Fusionner ou supprimer les branches quand le travail est terminé

### Collaboration

1. **Pull avant push :** Toujours récupérer les derniers changements avant de pousser votre travail
2. **Utiliser les pull requests :** Réviser le code avant de fusionner dans les branches principales
3. **Communiquer :** Utiliser les messages de commit et les descriptions de PR pour expliquer vos changements

### Pièges Courants à Éviter

1. **Ne pas committer d'informations sensibles :** Utiliser `.gitignore` pour les secrets, clés API, etc.
2. **Ne pas travailler directement sur main :** Toujours utiliser des branches de fonctionnalité

## Référence Rapide

| Commande | Description |
|----------|-------------|
| `git clone <url>` | Télécharger un dépôt |
| `git status` | Vérifier l'état du répertoire de travail |
| `git add <fichier>` | Stagé les changements |
| `git commit -m "<message>"` | Créer un commit |
| `git push` | Télécharger les changements |
| `git pull` | Télécharger et fusionner les changements |
| `git checkout <branche>` | Basculer vers une branche |
| `git checkout -b <branche>` | Créer et basculer vers une branche |
| `git merge <branche>` | Fusionner une branche |
| `git log` | Voir l'historique des commits |
| `git stash` | Sauvegarder temporairement les changements |

## Pour aller plus loin

- `git help <commande>` - Obtenir de l'aide pour une commande spécifique
- `git <commande> --help` - Ouvrir la page de manuel pour une commande
- [Documentation Git](https://git-scm.com/doc) - Documentation officielle
- [Livre Pro Git](https://git-scm.com/book) - Guide complet gratuit