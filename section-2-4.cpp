#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

using namespace std;

class RPS {
   private:
    mt19937 generator;
    enum class ACTION { ROCK = 0, PAPER, SCISSORS };

    static constexpr int NUMBER_OF_ACTIONS = 3;

    array<double, NUMBER_OF_ACTIONS> regret_sum = {0};
    array<double, NUMBER_OF_ACTIONS> strategy = {0};
    array<double, NUMBER_OF_ACTIONS> strategy_sum = {0};
    array<double, NUMBER_OF_ACTIONS> opponents_strategy = {0.4, 0.3, 0.3};

    array<double, NUMBER_OF_ACTIONS> get_strategy() {
        double sum = 0;

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            strategy[a] = regret_sum[a] > 0 ? regret_sum[a] : 0;
            sum += strategy[a];
        }

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            if (sum > 0)
                strategy[a] /= sum;
            else
                strategy[a] = 1.0 / NUMBER_OF_ACTIONS;

            strategy_sum[a] += strategy[a];
        }

        return strategy;
    }

    ACTION get_action(array<double, NUMBER_OF_ACTIONS> strategy) {
        discrete_distribution<> distribution(strategy.begin(), strategy.end());

        int action_index = distribution(generator);
        return static_cast<ACTION>(action_index);
    }

    void normalize(array<double, NUMBER_OF_ACTIONS>& array) {
        double sum = 0;

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) sum += array[a];

        for (int a = 0; a < NUMBER_OF_ACTIONS; a++) {
            if (sum > 0)
                array[a] = array[a] / sum;
            else
                array[a] = 1.0 / NUMBER_OF_ACTIONS;
        }
    }

   public:
    RPS() : generator(0) {}

    array<double, NUMBER_OF_ACTIONS> train(int iterations) {
        array<double, NUMBER_OF_ACTIONS> action_utility;

        for (int i = 0; i < iterations; i++) {
            array<double, NUMBER_OF_ACTIONS> strategy = get_strategy();
            ACTION my_action = get_action(strategy);
            ACTION opponents_action = get_action(opponents_strategy);

            action_utility[(int)opponents_action] = 0;
            action_utility[(int)opponents_action == NUMBER_OF_ACTIONS - 1 ? 0 : (int)opponents_action + 1] = 1;
            action_utility[(int)opponents_action == 0 ? NUMBER_OF_ACTIONS - 1 : (int)opponents_action - 1] = -1;

            for (int a = 0; a < NUMBER_OF_ACTIONS; a++)
                regret_sum[a] += action_utility[a] - action_utility[(int)my_action];
        }

        normalize(strategy_sum);
        return strategy_sum;
    }
};

int main() {
    RPS rps;
    auto result = rps.train(100000);

    cout << fixed << setprecision(2);
    for (int i = 0; i < result.size(); i++) cout << result[i] << endl;

    return 0;
}
