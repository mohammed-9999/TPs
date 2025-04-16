import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Exercice 1: Initialisation de l'environnement et des structures de données
# Initialiser l'environnement Taxi-v3
env = gym.make("Taxi-v3")

# Nombre d'états et d'actions
state_size = env.observation_space.n
action_size = env.action_space.n

print(f"Nombre d'états: {state_size}")
print(f"Nombre d'actions: {action_size}")

# Créer une table de politique où chaque état a une probabilité égale pour chaque action
policy_table = np.ones((state_size, action_size)) / action_size

# Créer une table de valeurs initialisée à zéro
value_table = np.zeros(state_size)

# Afficher les premières lignes de policy_table et value_table
print("\nPremières lignes de policy_table:")
print(policy_table[:5])
print("\nPremières lignes de value_table:")
print(value_table[:5])


# Exercice 2: Exploration et collecte d'épisodes
def collect_episodes(env, policy, num_episodes=20, epsilon=0.1):
    """
    Fait exécuter un agent avec politique epsilon-greedy dans l'environnement
    pendant 'num_episodes' épisodes et collecte les états, actions et récompenses.

    Args:
        env: L'environnement Gymnasium
        policy: Table de politique
        num_episodes: Nombre d'épisodes à collecter
        epsilon: Probabilité d'exploration
    """
    episode_states = []
    episode_actions = []
    episode_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        states = []
        actions = []
        rewards = []
        done = False

        while not done:
            # Epsilon-greedy pour encourager l'exploration
            if np.random.random() < epsilon:
                action = np.random.choice(action_size)  # Action aléatoire
            else:
                # Sélectionner l'action en fonction de la politique
                action_probs = policy[state]
                if np.sum(action_probs) <= 0:  # Vérification de sécurité
                    action = np.random.choice(action_size)
                else:
                    action = np.random.choice(len(action_probs), p=action_probs)

            # Exécuter l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Stocker état, action et récompense
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Passer à l'état suivant
            state = next_state

        episode_states.append(states)
        episode_actions.append(actions)
        episode_rewards.append(rewards)

        print(f"Épisode terminé avec une récompense totale de {sum(rewards)}")

    return episode_states, episode_actions, episode_rewards


# Exercice 3: Mise à jour de la politique avec PPO
def compute_discounted_rewards(rewards, gamma=0.99):
    """Calcule les récompenses cumulées (discounted rewards)."""
    discounted_rewards = []
    for episode_rewards in rewards:
        discounted = []
        running_add = 0
        for r in reversed(episode_rewards):
            running_add = r + gamma * running_add
            discounted.insert(0, running_add)
        discounted_rewards.append(discounted)
    return discounted_rewards


def compute_advantages(discounted_rewards, value_table, states, gamma=0.99):
    """Calcule les avantages A(t) = R(t) - V(s(t))."""
    advantages = []
    for episode_rewards, episode_states in zip(discounted_rewards, states):
        episode_advantages = []
        for r, s in zip(episode_rewards, episode_states):
            advantage = r - value_table[s]
            episode_advantages.append(advantage)
        advantages.append(episode_advantages)
    return advantages


def update_policy_ppo(policy_table, advantages, states, actions, lr_policy=0.1, epsilon=0.2):
    """Mise à jour de la politique avec l'algorithme PPO et clipping améliorée."""
    for episode_states, episode_actions, episode_advantages in zip(states, actions, advantages):
        for state, action, advantage in zip(episode_states, episode_actions, episode_advantages):
            old_policy = policy_table[state].copy()

            # Approche basée sur les logits pour une mise à jour plus stable
            logits = np.log(old_policy + 1e-10)

            # Mettre à jour le logit de l'action choisie
            update_factor = lr_policy * advantage

            # Limiter l'amplitude des mises à jour pour éviter des changements trop brusques
            update_factor = np.clip(update_factor, -1.0, 1.0)

            # Appliquer la mise à jour
            logits[action] += update_factor

            # Convertir les logits en probabilités avec softmax
            exp_logits = np.exp(logits - np.max(logits))  # Soustraire le max pour stabilité numérique
            policy_table[state] = exp_logits / np.sum(exp_logits)

    return policy_table


def update_value_table(value_table, discounted_rewards, states, lr_value=0.1):
    """Mise à jour de la fonction de valeur V(s)."""
    for episode_rewards, episode_states in zip(discounted_rewards, states):
        for r, s in zip(episode_rewards, episode_states):
            # Mise à jour plus progressive pour éviter les oscillations
            value_table[s] += lr_value * (r - value_table[s])
    return value_table


def train_ppo(env, num_iterations=100, num_episodes_per_iteration=20, gamma=0.99,
              lr_policy=0.1, lr_value=0.1, epsilon=0.2, exploration_eps=0.1):
    """Fonction principale d'entraînement avec PPO améliorée."""
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Initialisation des tables
    policy_table = np.ones((state_size, action_size)) / action_size
    value_table = np.zeros(state_size)

    all_rewards = []
    average_rewards = []
    best_avg_reward = -float('inf')
    best_policy = policy_table.copy()
    best_value = value_table.copy()

    # Réduction progressive du taux d'apprentissage et d'exploration
    initial_lr_policy, initial_lr_value = lr_policy, lr_value
    initial_exploration_eps = exploration_eps

    for iteration in range(num_iterations):
        print(f"\nItération {iteration + 1}/{num_iterations}")

        # Réduire progressivement les taux d'apprentissage et d'exploration
        current_lr_policy = initial_lr_policy * (1 - iteration / num_iterations)
        current_lr_value = initial_lr_value * (1 - iteration / num_iterations)
        current_exploration_eps = initial_exploration_eps * (1 - iteration / num_iterations)

        # Collecte d'épisodes avec exploration epsilon-greedy
        episode_states, episode_actions, episode_rewards = collect_episodes(
            env, policy_table, num_episodes_per_iteration, current_exploration_eps)

        # Calcul des récompenses cumulées
        discounted_rewards = compute_discounted_rewards(episode_rewards, gamma)

        # Calcul des avantages
        advantages = compute_advantages(discounted_rewards, value_table, episode_states, gamma)

        # Mise à jour de la politique avec PPO améliorée
        policy_table = update_policy_ppo(policy_table, advantages, episode_states,
                                         episode_actions, current_lr_policy, epsilon)

        # Mise à jour de la fonction de valeur
        value_table = update_value_table(value_table, discounted_rewards,
                                         episode_states, current_lr_value)

        # Stocker les récompenses moyennes
        episode_total_rewards = [sum(rewards) for rewards in episode_rewards]
        all_rewards.extend(episode_total_rewards)
        current_avg_reward = np.mean(episode_total_rewards)
        average_rewards.append(current_avg_reward)

        # Sauvegarder la meilleure politique
        if current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            best_policy = policy_table.copy()
            best_value = value_table.copy()

        print(f"Récompense moyenne pour l'itération {iteration + 1}: {current_avg_reward}")
        print(f"Meilleure récompense moyenne jusqu'à présent: {best_avg_reward}")
        print(f"Taux d'apprentissage: {current_lr_policy:.4f}, Exploration: {current_exploration_eps:.4f}")

    print(f"\nEntraînement terminé. Meilleure récompense moyenne: {best_avg_reward}")
    return best_policy, best_value, all_rewards, average_rewards


# Exercice 4: Évaluation de l'agent après entraînement
def evaluate_agent(env, policy_table, num_eval_episodes=20):
    """Teste l'agent entraîné pendant 'num_eval_episodes' épisodes."""
    total_rewards = []
    steps_taken = []

    for ep in range(num_eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 200:  # Limite le nombre d'étapes
            # Sélectionner l'action en fonction de la politique
            action_probs = policy_table[state]
            action = np.argmax(action_probs)  # Agent entraîné: prendre l'action avec la probabilité la plus élevée

            # Exécuter l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Accumuler la récompense et compter les étapes
            total_reward += reward
            steps += 1

            # Passer à l'état suivant
            state = next_state

        total_rewards.append(total_reward)
        steps_taken.append(steps)
        print(f"Évaluation - Épisode {ep + 1}/{num_eval_episodes}: Récompense = {total_reward}, Étapes = {steps}")

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_taken)
    print(f"\nRécompense moyenne sur {num_eval_episodes} épisodes d'évaluation: {avg_reward}")
    print(f"Nombre moyen d'étapes: {avg_steps}")

    return total_rewards, avg_reward


def visualize_taxi(env, policy_table, episodes=3, max_steps=100, sleep_time=0.5):
    """
    Visualise le comportement du taxi selon la politique apprise.

    Args:
        env: L'environnement Taxi-v3
        policy_table: La table de politique entraînée
        episodes: Nombre d'épisodes à visualiser
        max_steps: Nombre maximum d'étapes par épisode
        sleep_time: Temps d'attente entre chaque action pour la visualisation
    """
    actions_desc = {
        0: "Sud",
        1: "Nord",
        2: "Est",
        3: "Ouest",
        4: "Prise",
        5: "Dépôt"
    }

    for episode in range(episodes):
        state, _ = env.reset()
        
        print(f"\nÉpisode {episode + 1}/{episodes}")
        env.render()
        total_reward = 0
        step_count = 0

        for step in range(max_steps):
            step_count += 1

            # Sélectionner l'action selon la politique (prendre l'action avec la plus haute probabilité)
            action = np.argmax(policy_table[state])

            print(f"État: {state}, Action: {action} ({actions_desc[action]})")

            # Exécuter l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Afficher l'environnement après l'action
            env.render()
            total_reward += reward

            # Attendre pour permettre la visualisation
            time.sleep(sleep_time)

            # Mettre à jour l'état
            state = next_state

            if done:
                print(f"Épisode terminé en {step + 1} étapes avec récompense totale: {total_reward}")
                break

        if not done:
            print(f"Nombre maximum d'étapes atteint. Récompense totale: {total_reward}")

    print("Visualisation terminée.")


# Paramètres d'entraînement améliorés
gamma = 0.99
lr_policy = 0.05  # Taux d'apprentissage plus faible pour plus de stabilité
lr_value = 0.1
epsilon = 0.2
exploration_eps = 0.2  # Plus d'exploration
num_iterations = 500  # Plus d'itérations
num_episodes_per_iteration = 30  # Episodes par itération

# Entraînement PPO amélioré
print("\nDébut de l'entraînement PPO amélioré:")
policy_table, value_table, all_rewards, average_rewards = train_ppo(
    env,
    num_iterations=num_iterations,
    num_episodes_per_iteration=num_episodes_per_iteration,
    gamma=gamma,
    lr_policy=lr_policy,
    lr_value=lr_value,
    epsilon=epsilon,
    exploration_eps=exploration_eps
)

# Affichage des résultats d'entraînement
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(all_rewards)
plt.title('Récompenses par épisode')
plt.xlabel('Épisode')
plt.ylabel('Récompense totale')

plt.subplot(1, 2, 2)
plt.plot(average_rewards)
plt.title('Récompense moyenne par itération')
plt.xlabel('Itération')
plt.ylabel('Récompense moyenne')
plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

# Évaluation de l'agent après entraînement
print("\nÉvaluation de l'agent après entraînement:")
eval_rewards, avg_eval_reward = evaluate_agent(env, policy_table, num_eval_episodes=10)

# Visualisation de l'évaluation
plt.figure(figsize=(10, 5))
plt.bar(range(len(eval_rewards)), eval_rewards)
plt.axhline(y=avg_eval_reward, color='r', linestyle='-', label=f'Moyenne: {avg_eval_reward:.2f}')
plt.title('Récompenses pendant l\'évaluation')
plt.xlabel('Épisode')
plt.ylabel('Récompense totale')
plt.legend()
plt.savefig('evaluation_results.png')
plt.show()

# Visualisation du comportement du taxi
print("\nVisualisation du comportement du taxi:")
visualize_taxi(env=gym.make("Taxi-v3", render_mode="human"),
               policy_table=policy_table,
               episodes=3,
               max_steps=100,  # Plus d'étapes pour permettre de terminer l'épisode
               sleep_time=0.5)

# Fermer l'environnement
env.close()