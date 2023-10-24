"""
Bandit Algorithms

Overview:
This Python code defines and demonstrates the workings of two different multi-armed bandit (MAB) algorithms: Epsilon-Greedy and Thompson Sampling. 
The code includes classes for bandits, visualization, and the main script for running the experiments and comparing the two algorithms.

Author: Naira Maria Barseghyan


"""
############################### LOGGER
from abc import ABC, abstractmethod
from typing import Self
from logs import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


EPS = 0.1
NumberOfTrials = 20000
Bandit_Rewards = [1, 2, 3, 4]

class Bandit(ABC):
    
    """
    Abstract base class for bandit algorithms.

    Methods:
    - __init__(self, true_mean): Initialize the bandit with a true mean value.
    - __repr__(self): Provide a string representation of the bandit.
    - pull(self): Simulate pulling an arm of the bandit.
    - update(self, x): Update the bandit based on an observed reward.
    - experiment(self): Conduct an experiment with the bandit.
    - report(self): Report results, such as storing data and printing statistics.

    Attributes:
    - true_mean: The true mean value of the bandit.
    """

    @abstractmethod
    def __init__(self, true_mean):
        self.true_mean = true_mean

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass
    
    @abstractmethod
    def update(self, x):
        pass
        
        
    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass
    

#--------------------------------------#

class Visualization():
    
    """
    Class for visualizing bandit distributions and experiment results.

    Methods:
    - plot1(self, trial): Plot bandit distributions after a specified number of trials.
    - plot2(self, epsilon_greedy_rewards, thompson_rewards): Plot cumulative rewards comparison.

    Attributes:
    - bandits: A list of bandits to visualize.
    - num_trials: The number of trials for the experiment.
    """
    
    def __init__(self, bandits, num_trials):
        self.bandits = bandits
        self.num_trials = num_trials
    
    def plot1(self, trial):
        
        """
        Plot bandit distributions after a specified number of trials.

        Args:
        - trial: The trial number at which to plot the distributions.
        """
        
        x = np.linspace(-3, 6, 200)
        for b in self.bandits:
            y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
        plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

        
    def plot2(self, epsilon_greedy_rewards, thompson_rewards):
        
        """
        Plot cumulative rewards comparison between two bandit algorithms.

        Args:
        - epsilon_greedy_rewards: Cumulative rewards for the Epsilon-Greedy algorithm.
        - thompson_rewards: Cumulative rewards for the Thompson Sampling algorithm.
        """
        
        plt.plot(epsilon_greedy_rewards, label='Epsilon-Greedy')
        plt.plot(thompson_rewards, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Rewards Comparison")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()
        


class EpsilonGreedy(Bandit):
    
    """
    Bandit class implementing the Epsilon-Greedy algorithm.

    Methods:
    - __init__(self, true_mean, epsilon=EPS): Initialize the Epsilon-Greedy bandit.
    - pull(self): Simulate pulling an arm of the bandit.
    - update(self, x): Update the bandit's estimates based on an observed reward.
    - experiment(self, bandit_rewards, num_trials): Conduct an experiment with Epsilon-Greedy.
    - report(self): Report the results of the experiment.

    Attributes:
    - true_mean: The true mean value of the bandit.
    - epsilon: The exploration parameter (default: 0.1).
    - p_estimate: Estimated mean of the bandit.
    - N: Number of pulls.
    - m: Mean estimate.
    - lambda_: Precision parameter.
    """
    
    def __init__(self, true_mean, epsilon = EPS):
        super().__init__(true_mean)
        self.p_estimate = 0
        self.N = 0
        self.m = 0 
        self.lambda_ = 1
        
    def __repr__(self):
        return 'An Arm with {} Win Rate'.format(self.true_mean)
    
    def pull(self):
        """
        Simulate pulling an arm of the bandit.

        Returns:
        - The observed reward after pulling the arm.
        """
        
        return np.random.randn() + self.true_mean
    
    def update(self, x):
        """
        Update the bandit's estimates based on an observed reward.

        Args:
        - x: Observed reward.
        """
        
        self.N += 1.
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N
        
    def experiment(self, bandit_rewards = Bandit_Rewards, num_trials = NumberOfTrials):
        """
        Conduct an experiment with the Epsilon-Greedy bandit.

        Args:
        - bandit_rewards: List of true mean values for bandits.
        - num_trials: The number of trials in the experiment.

        Returns:
        - cumulative_average: Cumulative average rewards over trials.
        """
        
        bandits = [EpsilonGreedy(p) for p in bandit_rewards]
        
        data = np.empty(num_trials)
        means = np.array(bandit_rewards)
        true_best = np.argmax(means)
        count_suboptimal = 0
        
        sample_points = [10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 15000, 19999]
        vis = Visualization(bandits, num_trials)
        
        for i in range(num_trials):
            if np.random.random() < EPS / (i + 1):
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])
                
            x = bandits[j].pull()
            bandits[j].update(x)
            data[i] = x
            
            if j != true_best:
                count_suboptimal += 1

        cumulative_average = np.cumsum(data) / (np.arange(num_trials) + 1)
            
        return cumulative_average
    
    def report(self):
        """
        Report the results of the Epsilon-Greedy experiment.

        Outputs:
        - Data stored in CSV.
        """
        
        df = pd.DataFrame(columns = ['Bandit', 'Reward,', 'Algorithm})'])
        df = df._append({'Bandit' : self, 'Reward' : self.p_estimate, 'Algorithm' : "Epsilon Greedy"}, 
                ignore_index = True)
        df.to_csv('report_epsilon.csv', index=False)
        
        
        

class ThompsonSampling(Bandit):
    """
    Bandit class implementing the Thompson Sampling algorithm.

    Methods:
    - __init__(self, true_mean): Initialize the Thompson Sampling bandit.
    - pull(self): Simulate pulling an arm of the bandit.
    - sample(self): Sample from the bandit's distribution.
    - update(self, x): Update the bandit based on an observed reward.
    - experiment(self, bandit_rewards, num_trials): Conduct an experiment with Thompson Sampling.
    - report(self): Report the results of the experiment.

    Attributes:
    - true_mean: The true mean value of the bandit.
    - m: Mean estimate.
    - lambda_: Precision parameter.
    - tau: Precision parameter.
    - N: Number of pulls.
    """
    
    def __init__(self, true_mean):
        super().__init__(true_mean)
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        
    def __repr__(self):
        return 'An Arm with {} Win Rate'.format(self.true_mean)
    
    def pull(self):
        """
        Simulate pulling an arm of the bandit.

        Returns:
        - The observed reward after pulling the arm.
        """
        
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        """
        Sample from the bandit's distribution.

        Returns:
        - Sampled value from the distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m
    
    def update(self, x):
        """
        Update the bandit based on an observed reward.

        Args:
        - x: Observed reward.
        """
        
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        
    def experiment(self, bandit_reward = Bandit_Rewards, num_trials = NumberOfTrials):
        """
        Conduct an experiment with the Thompson Sampling bandit.

        Args:
        - bandit_rewards: List of true mean values for bandits.
        - num_trials: The number of trials in the experiment.

        Returns:
        - cumulative_average: Cumulative average rewards over trials.
        """
        
        bandits = [ThompsonSampling(m) for m in bandit_reward]
        rewards = np.empty(num_trials)
        
        sample_points = [10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 15000, 19999]
        vis = Visualization(bandits, num_trials)
        
        for i in range(num_trials):
            j = np.argmax([self.sample() for b in bandits])
            
            if i in sample_points:
                vis.plot1(i)
            
            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i] = x
            
                
        cumulative_average = np.cumsum(rewards) / (np.arange(num_trials) + 1)
        
        return cumulative_average
    
    def report(self):
        """
        Report the results of the Thompson Sampling experiment.

        Outputs:
        - Data stored in CSV.
        """
        average_reward = np.mean(self.experiment())
        
        df = pd.DataFrame(columns = ['Bandit', 'Reward,', 'Algorithm})'])
        df = df._append({'Bandit' : self, 'Average Reward' : average_reward, 'Algorithm' : "Thompson Sampling"}, 
                ignore_index = True)
        df.to_csv('report_thompson.csv', index=False)
        
        
        
        
    

def comparison(bandit1, bandit2, num_trials):
    """
    Compare the cumulative rewards of two bandit algorithms.

    Args:
    - bandit1: Cumulative rewards for the first bandit algorithm.
    - bandit2: Cumulative rewards for the second bandit algorithm.
    - num_trials: The number of trials in the experiment.
    """

    visualizer = Visualization([bandit1, bandit2], num_trials)
    visualizer.plot2(bandit1, bandit2)

if __name__ == '__main__':
    EPS = 0.1
    NUM_TRIALS = 20000
    Bandit_Rewards = [1, 2, 3, 4]

    epsilon_greedy = [EpsilonGreedy(i) for i in Bandit_Rewards]
    thompson_sampling = [ThompsonSampling(i) for i in Bandit_Rewards]
    
    epsilon_greedy_report = [i.report() for i in epsilon_greedy]
    thompson_sampling_report = [i.report() for i in thompson_sampling]
    
    epsilon_greedy_results = [i.experiment() for i in epsilon_greedy]
    thompson_sampling_results = [i.experiment() for i in thompson_sampling]

    for i in range(len(epsilon_greedy_results)):
        comparison(epsilon_greedy_results[i], thompson_sampling_results[i], NUM_TRIALS)

    
    logger.debug("debug message")
    logger.info("info message")
    #logger.warning("warning message")
    #logger.error("error message")
    #logger.critical("critical message")
