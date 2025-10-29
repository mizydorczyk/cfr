#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace std;

class Player {
   public:
    vector<double> regret_sum;
    vector<double> strategy;
    int num_actions;

    Player(int num_actions) : regret_sum(num_actions, 0.0), strategy(num_actions, 0.0), num_actions(num_actions) {}

    vector<double> get_strategy() {
        double sum = 0.0;

        for (int a = 0; a < num_actions; a++) {
            strategy[a] = regret_sum[a] > 0 ? regret_sum[a] : 0;
            sum += strategy[a];
        }

        for (int a = 0; a < num_actions; a++) {
            if (sum > 0)
                strategy[a] /= sum;
            else
                strategy[a] = 1.0 / num_actions;
        }

        return strategy;
    }
};

class ColonelBlotto {
   private:
    mt19937 generator;
    vector<vector<int>> all_actions;
    int num_actions;

    vector<vector<int>> get_available_actions(int n, int s) {
        vector<vector<int>> all_actions;
        vector<int> current(n, 0);

        function<void(int, int)> generate = [&](int idx, int remaining) {
            if (idx == n - 1) {
                current[idx] = remaining;
                all_actions.push_back(current);
                return;
            }
            for (int soldiers = 0; soldiers <= remaining; soldiers++) {
                current[idx] = soldiers;
                generate(idx + 1, remaining - soldiers);
            }
        };

        generate(0, s);

        return all_actions;
    }

    int get_action(const vector<double>& strategy) {
        discrete_distribution<> distribution(strategy.begin(), strategy.end());
        return distribution(generator);
    }

    vector<int> calculate_actions_utility(const int opponent_action) {
        vector<int> utility(num_actions, 0);

        for (int a = 0; a < num_actions; a++) {
            int score = 0;

            for (int i = 0; i < all_actions[a].size(); i++) {
                if (all_actions[a][i] > all_actions[opponent_action][i])
                    score++;
                else if (all_actions[a][i] < all_actions[opponent_action][i])
                    score--;
            }

            utility[a] = score;
        }

        return utility;
    }

    vector<double> get_average_strategy(const vector<double>& strategy_sum) {
        vector<double> avg(num_actions, 0.0);
        double sum = accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0);
        for (int a = 0; a < num_actions; a++) avg[a] = (sum > 0) ? strategy_sum[a] / sum : 1.0 / num_actions;
        return avg;
    }

   public:
    ColonelBlotto(int n, int s) : generator(0) {
        all_actions = get_available_actions(n, s);
        num_actions = all_actions.size();
    }

    vector<double> train(int iterations) {
        Player player1(num_actions);
        Player player2(num_actions);
        vector<double> strategy_sum(num_actions, 0.0);

        for (int i = 0; i < iterations; i++) {
            auto strategy1 = player1.get_strategy();
            auto strategy2 = player2.get_strategy();

            for (int a = 0; a < num_actions; a++) strategy_sum[a] += strategy1[a];

            int action1 = get_action(strategy1);
            int action2 = get_action(strategy2);

            vector<int> u1 = calculate_actions_utility(action2);
            vector<int> u2 = calculate_actions_utility(action1);

            for (int a = 0; a < num_actions; a++) {
                player1.regret_sum[a] += u1[a] - u1[action1];
                player2.regret_sum[a] += u2[a] - u2[action2];
            }
        }

        return get_average_strategy(strategy_sum);
    }

    const vector<vector<int>>& get_all_actions() const { return all_actions; }
};

int main() {
    const int num_battlefields = 3;
    const int num_soldiers = 5;

    ColonelBlotto solver = ColonelBlotto(num_battlefields, num_soldiers);
    auto result = solver.train(1'000'000);

    const auto& actions = solver.get_all_actions();

    cout << fixed << setprecision(2);
    for (size_t i = 0; i < actions.size(); i++) {
        cout << "(";
        for (size_t j = 0; j < actions[i].size(); j++) {
            cout << actions[i][j];
            if (j + 1 < actions[i].size()) cout << ", ";
        }
        cout << "): " << result[i] * 100 << "%" << endl;
    }

    return 0;
}
