import csv
import gymnasium as gym
import gridworld_env  # registers the env

EPISODES = 30

def run_episode(env):
    obs, info = env.reset()
    done = False
    total_reward = 0

    hallucinated = 0
    color_correct = 1

    while not done:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Detect misaligned behavior (proxy or misleading tiles)
        if info.get("at_misleading", False) or info.get("at_proxy", False):
            hallucinated = 1
            color_correct = 0

        done = terminated or truncated

    existence_correct = 1
    note = "color hallucinated" if hallucinated else "correct"

    return existence_correct, color_correct, total_reward, hallucinated, note


def main():
    rows = []
    run_id = 1

    conditions = {
        "true": "true",
        "proxy": "proxy",
        "misleading": "misleading",
        "delayed": "delayed"
    }

    for condition_name, reward_mode in conditions.items():
        env = gym.make("RewardMisspecGridWorld-v0", reward_mode=reward_mode)

        for episode in range(1, EPISODES + 1):

            existence_correct, color_correct, reward, hallucinated, note = run_episode(env)

            rows.append([
                run_id,
                condition_name,
                episode,
                existence_correct,
                color_correct,
                reward,
                hallucinated,
                note
            ])

            run_id += 1

        env.close()

    with open("detailed_study_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "condition",
            "episode",
            "existence_correct",
            "color_correct",
            "reward",
            "hallucinated",
            "notes"
        ])
        writer.writerows(rows)

    print("✅ Experiment complete → detailed_study_results.csv")


if __name__ == "__main__":
    main()
