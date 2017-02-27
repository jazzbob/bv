import numpy as np
from scipy.stats import beta, bernoulli
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import os
import seaborn as sns


class Planner:
    def __init__(self):
        self.nr_plans = n_choices
        self.nr_actions = 10
        self.params = [-1, 1]

    def plan(self):
        return [self.get_random_plan(self.nr_actions) for _ in range(0, self.nr_plans)]

    def get_random_plan(self, nr_actions):
        return [self.get_random_action() for _ in range(0, nr_actions)]

    def get_random_action(self):
        return np.random.choice(self.params, size=2)


class World:
    def __init__(self, x_dim=10, y_dim=10):
        self.grid = np.random.rand(x_dim, y_dim)
        self.init_grid()
        self.agent = Agent()

    def init_grid(self):
        for x in range(len(self.grid)):
            for y in range(len(self.grid)):
                self.grid[x, y] = np.random.choice([0, -1], p=[0.8, 0.2])

    def update(self, action):
        self.agent.update(action)
        self.agent.pos = np.clip(self.agent.pos, 0, 9)
        reward = self.grid[self.agent.pos[0], self.agent.pos[1]]
        return reward

    def execute_plan(self, plan):
        r = [self.update(action) for action in plan]
        return np.sum(r)


class Agent:
    def __init__(self):
        self.pos = (0, 0)
        self.p_fail = np.random.uniform(0, 1)

    def update(self, action):
        if np.random.uniform(0, 1) <= self.p_fail:
            action *= -1
        self.pos += action


class Choice:
    def __init__(self, plan, p_req):
        self.plan = plan
        self.p_req = p_req
        self.successes = 1
        self.failures = 1
        self.bayes_factor = 1.

    def sample_value(self):
        return beta(self.successes, self.failures).rvs()

    def get_inverse_prior_odds(self):
        p_violated_prior = beta(1, 1).cdf(self.p_req)
        p_sat_prior = 1 - p_violated_prior
        return p_violated_prior / p_sat_prior

    def update_bayes_factor(self):
        p_violated = beta(self.successes, self.failures).cdf(self.p_req)
        p_sat = 1 - p_violated
        self.bayes_factor = self.get_inverse_prior_odds() * (p_sat / p_violated)
		
    def get_p_sat(self):
        p_violated = beta(self.successes, self.failures).cdf(self.p_req)
        p_sat = 1 - p_violated
        return p_sat


class Checker:
    def __init__(self):
        self.r_req = r_req
        self.p_req = p_req
        self.c_req = c_req
        self.t = 10
        self.choices = []
        self.simulation_runs = 0

    def check(self, world, plans, parameters):
        self.choices = [Choice(plan, self.p_req) for plan in plans]

        while True:
            choice = parameters.selection_method(self)

            world_copy = copy.deepcopy(world)
            # set parameter wrt. belief
            p_belief = parameters.belief_method(self, parameters.observations)
            world_copy.agent.p_fail = p_belief
            r = world_copy.execute_plan(choice.plan)
            self.simulation_runs += 1
            if self.simulation_runs % 10 == 0:
                plt.pause(0.001)

            if r >= self.r_req:
                choice.successes += 1
            else:
                choice.failures += 1

            if not np.isclose(choice.get_p_sat(), 1 - beta(choice.successes, choice.failures).cdf(self.p_req)):
                print(choice.get_p_sat(), 1 - beta(choice.successes, choice.failures).cdf(self.p_req))

            if choice.get_p_sat() >= self.c_req:
                print(choice.get_p_sat())
                return choice.plan
            if choice.get_p_sat() < 1 - self.c_req:
                self.choices.remove(choice)
            if not self.choices:
                print('reject')
                return []

    def ts_selection(self):
        p_safe = [choice.sample_value() for choice in self.choices]
        choice_index = np.argmax(p_safe)
        return self.choices[choice_index]

    def bayes_factor_selection(self):
        p_safe = [choice.bayes_factor for choice in self.choices]
        choice_index = np.argmax(p_safe)
        return self.choices[choice_index]

    def qos_selection(self):
        p_safe = [choice.get_p_sat() for choice in self.choices]
        choice_index = np.argmax(p_safe)
        return self.choices[choice_index]

    def uniform_selection(self):
        return np.random.choice(self.choices)

    def epsilon_greedy_selection(self):
        if np.random.uniform() < 0.1:
            return self.uniform_selection()
        p_safe = [choice.successes / (choice.successes + choice.failures) for choice in self.choices]
        choice_index = np.argmax(p_safe)
        return self.choices[choice_index]

    def ts_belief(self, observations):
        s = sum(observations)
        f = len(observations) - s
        return beta(s + 1, f + 1).rvs()

    def mle_belief(self, observations):
        s = sum(observations)
        return s / len(observations)


def evaluate_plan_percentile(world, plan, percentile, n_runs):
    results = []
    for _ in range(0, n_runs):
        world_copy = copy.deepcopy(world)
        r = world_copy.execute_plan(plan)
        results.append(r)
    return np.percentile(results, q=percentile)


def run_checker(world, plans, parameters):
    checker = Checker()
    checked_plan = checker.check(world, plans, parameters)
    if checked_plan:
        type_1_errors[parameters.label].append(False)
        accept_trials[parameters.label].append(checker.simulation_runs)
        result_tail = evaluate_plan_percentile(world, checked_plan, (1 - p_req) * 100, 10000)
        is_type_2_error = result_tail < checker.r_req
        type_2_errors[parameters.label].append(is_type_2_error)
    else:
        type_2_errors[parameters.label].append(False)
        reject_trials[parameters.label].append(checker.simulation_runs)
        for plan in plans:
            result_tail = evaluate_plan_percentile(world, plan, (1 - p_req) * 100, 10000)
            if result_tail >= checker.r_req:
                type_1_errors[parameters.label].append(True)
                return
        type_1_errors[parameters.label].append(False)


class Parameters:
    def __init__(self, label, selection_method, belief_method, observations):
        self.label = label
        self.selection_method = selection_method
        self.belief_method = belief_method
        self.observations = observations


def run_experiment():
    world = World()
    planner = Planner()
    plans = planner.plan()

    for o in n_observations:
        observations = bernoulli(world.agent.p_fail).rvs(size=o)
        run_checker(world, plans,
                    Parameters('bayes_{}'.format(o), Checker.ts_selection, Checker.ts_belief, observations))
        # run_checker(world, plans,
        #            Parameters('qos_{}'.format(o), Checker.qos_selection, Checker.ts_belief, observations))
        run_checker(world, plans,
                    Parameters('mle_{}'.format(o), Checker.ts_selection, Checker.mle_belief, observations))


def plot_results(experiment):
    x = range(experiment)
    error_bound = [data * (1 - c_req) for data in x]

    plt.figure(1)
    plt.clf()
    for key in sorted(type_1_errors.keys()):
        plt.plot(np.cumsum(type_1_errors[key]), label=key)
    plt.plot(x, error_bound, ls='--', label='{:.2} bound'.format(1 - c_req), color=(0, 0, 0, 1))
    plt.legend(loc='best', fancybox=False, framealpha=0.5)
    plt.suptitle('type 1 errors', fontsize=10)
    plt.savefig('{}/type_1_errors.png'.format(directory))

    plt.figure(2)
    plt.clf()
    for key in sorted(type_2_errors.keys()):
        plt.plot(np.cumsum(type_2_errors[key]), label=key)
    plt.plot(x, error_bound, ls='--', label='{:.2} bound'.format(1 - c_req), color=(0, 0, 0, 1))
    plt.legend(loc='best', fancybox=False, framealpha=0.5)
    plt.suptitle('type 2 errors', fontsize=10)
    plt.savefig('{}/type_2_errors.png'.format(directory))

    plt.figure(3)
    plt.clf()
    boxes = []
    labels = []
    for key in sorted(accept_trials.keys()):
        if accept_trials[key]:
            boxes.append(accept_trials[key])
            labels.append(key)
    if boxes:
        plt.boxplot(boxes, showfliers=False)
        plt.gca().set_xticklabels(labels)
    plt.suptitle('accept trials', fontsize=10)
    plt.savefig('{}/accept_trials.png'.format(directory))

    plt.figure(4)
    plt.clf()
    boxes = []
    labels = []
    for key in sorted(reject_trials.keys()):
        if reject_trials[key]:
            boxes.append(reject_trials[key])
            labels.append(key)
    if boxes:
        plt.boxplot(boxes, showfliers=False)
        plt.gca().set_xticklabels(labels)
    plt.suptitle('reject trials', fontsize=10)
    plt.savefig('{}/reject_trials.png'.format(directory))

    plt.pause(0.001)


directory = "plots/bv/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(directory):
    os.makedirs(directory)

np.random.seed(4242)
plt.ion()
plt.show()

r_req = -2
p_req = 0.9
c_req = 0.95
n_observations = [10]
n_choices = 1

type_1_errors = dict()
type_2_errors = dict()
accept_trials = dict()
reject_trials = dict()

for o in n_observations:
    type_1_errors['bayes_{}'.format(o)] = []
    type_2_errors['bayes_{}'.format(o)] = []
    accept_trials['bayes_{}'.format(o)] = []
    reject_trials['bayes_{}'.format(o)] = []

    '''
    type_1_errors['qos_{}'.format(o)] = []
    type_2_errors['qos_{}'.format(o)] = []
    accept_trials['qos_{}'.format(o)] = []
    reject_trials['qos_{}'.format(o)] = []
    '''

    type_1_errors['mle_{}'.format(o)] = []
    type_2_errors['mle_{}'.format(o)] = []
    accept_trials['mle_{}'.format(o)] = []
    reject_trials['mle_{}'.format(o)] = []

for i in range(0, 1000):
    print('experiment {}'.format(i))
    run_experiment()
    plot_results(i + 1)

plt.ioff()
plt.show()
