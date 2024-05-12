import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import entropy


class SupervisionHelper:
    def __init__(self, num_agents, episode_length, max_price, mape=True, progressive_punishment=False):
        """
        :param num_agents:
        :param episode_length:
        :param mape: Bool value indicating if the regression should return the Mean Absolute Percentage Error or the MAE.
        :param progressive_punishment: Bool value to activate a progressive punishment, that punish the agents harder,
        the closer they get to perfect predictability, controlled by parameter 'k' in calc_reward_factors().
        """
        self.num_agents = num_agents
        self.episode_id = 0
        self.step_id = 0
        self.episode_length = episode_length
        self.max_price = max_price
        self.violations_until_punishment = 0  # pick a number > 0 to enable sparse punishment
        self.violation_counter = 5
        # Create an empty DataFrame
        columns = ["episode_id", "step_id"] + [f"agent_{i}_price" for i in range(self.num_agents)] + [f"agent_{i}_class"
                                                                                                      for i in range(
                self.num_agents)]
        self.action_memory = pd.DataFrame(columns=columns)

        # Define classification input and target cols
        self.CLASSIFICATION_INPUT_COLS = columns[:-self.num_agents]
        self.CLASSIFICATION_TARGET_COLS = [f"agent_{i}_class" for i in range(self.num_agents)]

        # Define regression input and target rows
        self.REGRESSION_INPUT_COLS = self.CLASSIFICATION_INPUT_COLS
        self.REGRESSION_TARGET_COLS = [f"agent_{i}_price" for i in range(self.num_agents)]

        # set a flag for using Mean Absolute Percentage Error above Mean Absolute Error
        self.mape = mape

        # set a flag for using progressive punishment rather than linear punishment
        self.progressive_punishment = progressive_punishment

        # set bounds for linear conversion, first bound relates to the wanted and last bound to the unwanted behavior
        # i.e., a classification accuracy of 1 will result in a reward of 0
        self.classification_bounds = (0.67, 1)
        self.regression_mae_bounds = (0, 0.2)

    def report_prices(self, episode_id, step_id, agents):
        """
        collect prices from agents and store into memory for further evaluation
        :param episode_id: the environment's current epiode as int
        :param step_id: the environment's current step as int
        :param agents: a list of dicts containing information about the agents
        """
        # Save Episode and Step_ID
        self.episode_id = episode_id
        self.step_id = step_id

        # Extract prices from agent_info and create a new row
        prices = [agent['price'][0] for agent in agents]
        # check if agents are in the correct order
        assert [agent['id'] for agent in agents] == list(range(self.num_agents))

        # Classify prices
        if not self.action_memory.empty:
            last_prices = self.action_memory.iloc[-1][2:self.num_agents + 2]
            classifications = self._classify_prices(prices, last_prices)
        else:
            classifications = [0] * self.num_agents

        # Append the new row to the DataFrame
        new_row = [episode_id, step_id] + prices + classifications
        self.action_memory.loc[len(self.action_memory)] = new_row

    # Define a helper function to classify the values
    def _classify_prices(self, new_prices, last_prices):
        return [0 if new_price < last_price else 1 if new_price == last_price else 2 for new_price, last_price in
                zip(new_prices, last_prices)]

    def classification(self, min_steps_to_analyse=365, max_episodes_to_analyse=10):
        """
        Train a decision tree classifier and return classification accuracies.
        :param min_steps_to_analyse: Int value containing the min number of steps that have to pass until classification starts.
        :param max_episodes_to_analyse: Int value containing the num of past episodes used in the classification task.
        :return: Dictionary of classification accuracies for each target column.
        """
        if len(self.action_memory) > min_steps_to_analyse:
            # Determine window size to reduce runtime (episode length * number of episodes to check)
            max_window_size = min(len(self.action_memory), self.episode_length * max_episodes_to_analyse)

            # Prepare data
            X = self.action_memory.tail(max_window_size)[self.CLASSIFICATION_INPUT_COLS]
            y = self.action_memory.tail(max_window_size)[self.CLASSIFICATION_TARGET_COLS]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

            # Calculate classification accuracy for each target column
            accuracies = {}

            # Train a decision tree classifier for each target column
            for i, col in enumerate(self.CLASSIFICATION_TARGET_COLS):
                classifier = DecisionTreeClassifier()
                classifier.fit(X_train, y_train[col])
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test[col], y_pred)
                accuracies[f"agent_{i}_accuracy"] = accuracy

            return accuracies
        else:
            return {f"agent_{i}_accuracy": 0 for i in range(self.num_agents)}

    def regression(self, min_steps_to_analyse=365, max_episodes_to_analyse=10):
        """
        Train a decision tree regressor and return regression mean squared errors for prices.
        :param min_steps_to_analyse: Int value containing the min number of steps that must pass until regression starts.
        :param max_episodes_to_analyse: Int value containing the number of past episodes used in the regression task.
        :return: Mean squared error for price columns.
        """
        if len(self.action_memory) > min_steps_to_analyse:
            # Determine window size to reduce runtime (episode length * number of episodes to check)
            max_window_size = min(len(self.action_memory), self.episode_length * max_episodes_to_analyse)

            # Prepare data
            X = self.action_memory.tail(max_window_size)[self.REGRESSION_INPUT_COLS]
            y = self.action_memory.tail(max_window_size)[self.REGRESSION_TARGET_COLS]

            # Shift the target columns to represent future prices
            y = y.shift(-1)

            # Drop the last row since there is no future data for it
            X = X.iloc[:-1]
            y = y.iloc[:-1]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

            # Calculate mean squared error for price columns
            error_values = {}

            # Train a decision tree regressor for each price column
            for i, col in enumerate(self.REGRESSION_TARGET_COLS):
                regressor = DecisionTreeRegressor()
                regressor.fit(X_train, y_train[col])
                y_pred = regressor.predict(X_test)
                error_value = 0
                if not self.mape:
                    # mae calculation
                    error_value = mean_absolute_error(y_test[col], y_pred)
                else:
                    # MAPE calculation
                    error_value = min(mean_absolute_percentage_error(y_test[col], y_pred), 1) # TODO: UNSURE IF MAPE NEEDS BIGGER WINDOW SIZE
                error_values[f"agent_{i}_error_value"] = error_value

            return error_values
        else:
            return {f"agent_{i}_error_value": 999 if not self.mape else 1 for i in range(self.num_agents)}

    def calc_reward_factors(self, classification_accuracy=0, regression_error=0.2, k=2):
        """
        Train a decision tree regressor and return regression mean squared errors for prices.
        :param classification_accuracy: Float value containing the classification accuracy.
        :param regression_error: Float value containing the regression MAE value.
        :param k: Parameter to set the steepness of the progressive punishment curve.
        :return: the reward for one agent in float
        """
        # Define the weights for classification and regression
        regression_weight = 0.0
        classification_weight = 1.0

        # reward calculation through value interpolation; the last value sets the bounds and the order.
        classification_reward = np.interp(classification_accuracy, self.classification_bounds, (1, 0))
        if not self.mape:
            regression_reward = np.interp(regression_error, self.regression_mae_bounds, (0, 1))
        else:
            regression_reward = regression_error

        # Calculate the weighted average using np.average
        weighted_average_reward = np.average([classification_reward, regression_reward],
                                             weights=[classification_weight, regression_weight])

        if self.progressive_punishment:
            # remap the value to a curve that goes through (0|0) and (1|1), with its slope adjusted by 'k'
            weighted_average_reward = (k**weighted_average_reward - 1) / (k - 1)

        return weighted_average_reward

    def calc_reward_factors_time_series(self):
        """
        calculate a reward factor that rewards random behavior.
        Thus, we calculate
        autocorrelation: 1 is a perfect positive linear relationship, -1 is perfect negative linear relationship, and
                         0 indicates no linear relationship.
        volatility: A higher value indicates higher volatility or variability in the percentage changes.
                    A lower value suggests lower volatility or more stable percentage changes.
        entropy: Higher entropy values indicate a more disordered or uncertain distribution.
                 Lower values suggest a more concentrated or predictable distribution.
        :return: a list containing the rewards for each agent in float
        """
        def calculate_features(price_data):
            autocorr = price_data.autocorr()
            volatility = price_data.pct_change().std()
            price_distribution = np.histogram(price_data, bins='auto', density=True)[0]
            entropy_value = entropy(price_distribution)
            return autocorr, volatility, entropy_value

        # handle the case that the action memory is empty or hasn't run an episode yet
        if self.action_memory is None or len(self.action_memory) < self.episode_length:
            # Return 0 reward in the initial steps
            return [0] * self.num_agents

        rewards = []
        for agent_id in range(self.num_agents):
            column_name = f"agent_{agent_id}_price"
            price_data = pd.Series(self.action_memory[column_name])
            autocorr, volatility, entropy_value = calculate_features(price_data)

            # Reward calculation
            autocorr_reward = abs(1 - abs(autocorr))  # Reward inversely proportional to autocorrelation
            volatility_reward = volatility  # Reward inversely proportional to volatility
            entropy_reward = entropy_value  # Reward proportional to entropy

            rewards.append(autocorr_reward + entropy_reward + entropy_reward)

        return rewards

    def action_mask_random_agent(self, reward_factors, mode=0, min_steps_to_analyse=365):
        # If ALL agents have a supervision reward factor below X, punish them
        if len(self.action_memory) > min_steps_to_analyse:
            # check if all agents violate the law by acting predictable
            if all(factor < 0.5 for factor in reward_factors):
                self.violation_counter += 1

                # when the agents violate the law x times, do this
                if self.violations_until_punishment >= 5:
                    # reset counter
                    self.violation_counter = 0

                    # choose an agent to be punished
                    random_agent_id = np.random.randint(self.num_agents)
                    random_agents_last_action = (self.action_memory.iloc[-1][f"agent_{random_agent_id}_price"]
                                                 - self.action_memory.iloc[-2][f"agent_{random_agent_id}_price"])

                    # set a new action as punishment
                    new_action = 0
                    if mode == 0:
                        # same action but in the opposite pricing direction
                        new_action = random_agents_last_action * -1
                    if mode == 1:
                        # random action BUT the expected one
                        while True:
                            new_action = np.random.uniform(self.max_price)
                            if new_action != random_agents_last_action:
                                break
                    if mode == 2:
                        # keep price the same (basically ignoring the new action)
                        random_action = self.action_memory.iloc[-2][f"agent_{random_agent_id}_price"]

                    # write the manipulated action in the action dict
                    self.action_memory.iloc[-1][f"agent_{random_agent_id}_price"] = (
                            self.action_memory.iloc[-2][f"agent_{random_agent_id}_price"] + new_action)

        return self.action_memory.iloc[-1][[f"agent_{i}_price" for i in range(self.num_agents)]].values
