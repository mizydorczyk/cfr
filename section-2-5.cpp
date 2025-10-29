#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

using namespace std;

static constexpr int NUMBER_OF_ACTIONS = 3;

class Player {
   public:
    array<double, NUMBER_OF_ACTIONS> regret_sum = {0};
    array<double, NUMBER_OF_ACTIONS> strategy = {0};

    array<double, NUMBER_OF_ACTIONS> get_strategy() {
        double sum = 0.0;

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            strategy[a] = regret_sum[a] > 0 ? regret_sum[a] : 0;
            sum += strategy[a];
        }

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            if (sum > 0)
                strategy[a] /= sum;
            else
                strategy[a] = 1.0 / NUMBER_OF_ACTIONS;
        }

        return strategy;
    }
};

class RPS {
   private:
    mt19937 generator;
    enum class ACTION { ROCK = 0, PAPER, SCISSORS };

    ACTION get_action(array<double, NUMBER_OF_ACTIONS>& strategy) {
        discrete_distribution<> distribution(strategy.begin(), strategy.end());

        int action_index = distribution(generator);
        return static_cast<ACTION>(action_index);
    }

    array<double, NUMBER_OF_ACTIONS> calculate_actions_utility(ACTION opponent_action) {
        array<double, NUMBER_OF_ACTIONS> utility = {0};

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            if (a == (int)opponent_action) {
                // tie
                utility[a] = 0;
            } else if ((a + 1) % NUMBER_OF_ACTIONS == (int)opponent_action) {
                // lose
                utility[a] = -1;
            } else {
                // win
                utility[a] = 1;
            }
        }

        return utility;
    }

    array<double, NUMBER_OF_ACTIONS> get_average_strategy(array<double, NUMBER_OF_ACTIONS>& strategy_sum) {
        array<double, NUMBER_OF_ACTIONS> average_strategy;
        double sum = 0;

        for (auto s : strategy_sum) {
            sum += s;
        }

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            average_strategy[a] = sum > 0 ? strategy_sum[a] / sum : 1.0 / NUMBER_OF_ACTIONS;
        }

        return average_strategy;
    }

   public:
    RPS() : generator(0) {}

    array<double, NUMBER_OF_ACTIONS> train(int iterations) {
        Player player1;
        Player player2;

        array<double, NUMBER_OF_ACTIONS> strategy_sum1 = {0};

        for (int i = 0; i < iterations; i++) {
            auto strategy1 = player1.get_strategy();
            auto strategy2 = player2.get_strategy();

            for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
                strategy_sum1[a] += strategy1[a];
            }

            ACTION action1 = get_action(strategy1);
            ACTION action2 = get_action(strategy2);

            auto u1 = calculate_actions_utility(action2);
            auto u2 = calculate_actions_utility(action1);

            for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
                player1.regret_sum[a] += u1[a] - u1[(int)action1];
                player2.regret_sum[a] += u2[a] - u2[(int)action2];
            }
        }

        return get_average_strategy(strategy_sum1);
    }
};

int main() {
    RPS rps;
    auto result = rps.train(1000000);

    cout << fixed << setprecision(2);
    for (int i = 0; i < result.size(); i++) cout << result[i] << endl;

    return 0;
}
