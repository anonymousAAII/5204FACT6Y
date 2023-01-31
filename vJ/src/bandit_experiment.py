import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from auditing import OCEF

def bandit_experiment(arms, N, delta, alpha, epsilon):
    """
    Performs bandit experiments by running OCEF on the provided arms.
    Inputs:
        arms - List of arms. The first arm is assumed to be the baseline. Arms
               are defined here as the expectation of a Bernoulli distribution
               (and are thus floats between 0 and 1).
        N - Number of trials to perform
        delta - Confidence parameter for OCEF
        alpha - Conservative exploration parameter for OCEF
        epsilon - Envy parameter for OCEF
    Outputs:
        mean_duration - Average number of steps per trial
        mean_cost - Average auditing cost per trial
    """
    total_duration = 0
    total_cost = 0
    def reward_func(p):
        if p > np.random.rand():
            return 1
        else:
            return 0
    for _ in range(N):
        _, duration, cost = OCEF(arms, reward_func, delta, alpha, epsilon)
        total_duration += duration
        total_cost += cost
    mean_duration = total_duration / N
    mean_cost = total_cost / N
    return mean_duration, mean_cost

if __name__ == '__main__':
    np.random.seed(42)
    alphas = np.linspace(0.01, 0.5, 11)
    results = dict()
    delta = 0.05
    epsilon = 0.05
    N = 100
    # Create bandit scenarios
    scenarios = []
    scenarios.append(np.full(10, 0.3))
    scenarios[0][0] = 0.6
    scenarios.append(np.full(10, 0.3))
    scenarios[1][1] = 0.6
    scenarios.append(np.fromfunction(lambda k: 0.7 - 0.7 * (k / 10) ** 0.6, (10,)))
    scenarios.append(np.fromfunction(lambda k: 0.7 - 0.7 * (k / 10) ** 0.6, (10,)))
    scenarios[3][0] = scenarios[2][1]
    scenarios[3][1] = scenarios[2][0]
    scenario_dicts = []
    
    # Run scenarios
    for i, scenario in enumerate(scenarios):
        print(f'\nScenario {i+1}')
        d = dict()
        scenario_dicts.append(d)
        d['durations'] = []
        d['costs'] = []
        for alpha in tqdm(alphas):
            duration, cost = bandit_experiment(scenario, N, delta, alpha,
                                               epsilon)
            d['durations'].append(duration)
            d['costs'].append(cost)
    with open('../results/bandits', 'wb') as f:
        pickle.dump(scenario_dicts, f)
    
    # Plots
    for i, scenario in enumerate(scenario_dicts):
        plt.plot(alphas, scenario['durations'], label=i+1)
    plt.xlabel("alpha")
    plt.ylabel("duration")
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    for i, scenario in enumerate(scenario_dicts):
        plt.plot(alphas, scenario['costs'], label=i+1)
    plt.xlabel("alpha")
    plt.ylabel("costs")
    plt.legend()
    plt.show()
