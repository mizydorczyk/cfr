// Section 5.4
// Two-player Fixed-Strategy Iteration Counterfactual Regret Minimization (FSICFR) for the last round of Dudo.
// Exercise from "An Introduction to Counterfactual Regret Minimization" by Todd W. Neller and Marc Lanctot

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

const int NUM_SIDES = 6, NUM_ACTIONS = (2 * NUM_SIDES) + 1, DUDO = NUM_ACTIONS - 1;
const int CLAIM_NUM[] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
const int CLAIM_RANK[] = {2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1};

class Node {
   public:
    double utility, p_player, p_opponent;
    vector<double> regret_sum, strategy, strategy_sum;

    Node() {}

    Node(int num_actions) {
        regret_sum = vector<double>(num_actions, 0.0);
        strategy = vector<double>(num_actions, 0.0);
        strategy_sum = vector<double>(num_actions, 0.0);
    }

    // Get Dudo node current mixed strategy through regret-matching
    vector<double> get_strategy() {
        double sum = 0.0;

        for (int a = 0; a < strategy.size(); a++) {
            strategy[a] = regret_sum[a] > 0 ? regret_sum[a] : 0;
            sum += strategy[a];
        }

        for (int a = 0; a < strategy.size(); a++) {
            if (sum > 0)
                strategy[a] /= sum;
            else
                strategy[a] = 1.0 / strategy.size();

            strategy_sum[a] += p_player * strategy[a];
        }

        return strategy;
    }

    // Get Dudo node average mixed strategy
    vector<double> get_average_strategy() {
        vector<double> avg(strategy_sum.size(), 0.0);
        double sum = accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0);

        for (int a = 0; a < strategy_sum.size(); a++) {
            avg[a] = (sum > 0) ? strategy_sum[a] / sum : 1.0 / strategy_sum.size();
        }

        return avg;
    }
};

class DudoTrainer {
   private:
    vector<vector<Node>> response_nodes;
    vector<vector<Node>> claim_nodes;
    mt19937 generator{0};

    double get_dudo_utility(int my_roll, int claim_idx) {
        double total_utility = 0;
        int target_rank = CLAIM_RANK[claim_idx];
        int target_num = CLAIM_NUM[claim_idx];

        for (int opponents_roll = 1; opponents_roll <= 6; opponents_roll++) {
            int matches = 0;
            if (my_roll == target_rank || my_roll == 1) matches++;
            if (opponents_roll == target_rank || opponents_roll == 1) matches++;
            total_utility += (matches < target_num) ? 1.0 : -1.0;
        }
        return total_utility / 6.0;
    }

   public:
    DudoTrainer() {
        // response_nodes[last_claim_index][roll_index] -> {0: Dudo, 1: Continue}
        response_nodes = vector<vector<Node>>(NUM_ACTIONS, vector<Node>(NUM_SIDES + 1));
        for (int last_claim = 0; last_claim < NUM_ACTIONS; last_claim++) {
            for (int roll = 1; roll <= NUM_SIDES; roll++) {
                // If someone makes the very last possible claim (2x1) the next player must call {0: Dudo}
                int actions = (last_claim == NUM_ACTIONS - 1) ? 1 : 2;

                response_nodes[last_claim][roll] = Node(actions);
            };
        }

        // claim_nodes[last_claim_index][roll_index] -> [available claims]
        // If Continue is chosen, player picks a new claim greater than last claim
        claim_nodes = vector<vector<Node>>(NUM_ACTIONS, vector<Node>(NUM_SIDES + 1));
        for (int last_claim = 0; last_claim < NUM_ACTIONS - 1; last_claim++) {
            for (int roll = 1; roll <= NUM_SIDES; roll++) {
                int available_actions = (NUM_ACTIONS - 1) - last_claim;
                if (last_claim == 0) available_actions = NUM_ACTIONS - 1;  // Start of the game

                claim_nodes[last_claim][roll] = Node(available_actions);
            };
        }
    }

    void train(int iterations) {
        uniform_int_distribution<> dist(1, NUM_SIDES);
        vector<double> regret = vector<double>(NUM_ACTIONS);
        vector<int> roll_after_accepting_claim = vector<int>(NUM_ACTIONS);

        for (int iteration = 0; iteration < iterations; iteration++) {
            // Initialize rolls and starting probabilities
            for (int i = 0; i < roll_after_accepting_claim.size(); i++) {
                roll_after_accepting_claim[i] = dist(generator);
            }

            // Reset nodes' probabilities
            for (int claim = 0; claim < NUM_ACTIONS; claim++) {
                for (int roll = 1; roll <= NUM_SIDES; roll++) {
                    response_nodes[claim][roll].p_player = response_nodes[claim][roll].p_opponent = 0;
                    claim_nodes[claim][roll].p_player = claim_nodes[claim][roll].p_opponent = 0;
                }
            }

            int start_roll = roll_after_accepting_claim[0];
            claim_nodes[0][start_roll].p_player = 1.0;
            claim_nodes[0][start_roll].p_opponent = 1.0;

            // Accumulate realization weights forward
            for (int current_claim = 0; current_claim < NUM_ACTIONS - 1; current_claim++) {
                // Visit response nodes forward
                // "Opponent just claimed 'current_claim', do I Doubt or Continue?"
                if (current_claim > 0) {
                    Node& node = response_nodes[current_claim][roll_after_accepting_claim[current_claim]];

                    if (node.p_player > 0 || node.p_opponent > 0) {
                        vector<double> strategy = node.get_strategy();

                        if (strategy.size() > 1) {
                            Node& next_claim_node =
                                claim_nodes[current_claim][roll_after_accepting_claim[current_claim]];

                            next_claim_node.p_player += strategy[1] * node.p_player;
                            next_claim_node.p_opponent += node.p_opponent;
                        }
                    }
                }

                // Visit claim nodes forward
                // "I have accepted claim 'current_claim', what is my higher claim?"
                Node& node = claim_nodes[current_claim][roll_after_accepting_claim[current_claim]];
                if (node.p_player > 0 || node.p_opponent > 0) {
                    vector<double> action_probability = node.get_strategy();

                    for (int a = 0; a < action_probability.size(); a++) {
                        if (action_probability[a] > 0) {
                            int next_claim_idx = (current_claim == 0) ? (a + 1) : (current_claim + a + 1);

                            if (next_claim_idx < NUM_ACTIONS) {
                                Node& next_node =
                                    response_nodes[next_claim_idx][roll_after_accepting_claim[next_claim_idx]];

                                next_node.p_player += node.p_opponent;
                                next_node.p_opponent += action_probability[a] * node.p_player;
                            }
                        }
                    }
                }
            }

            // Backpropagate utilities, adjusting regrets and strategies
            for (int current_claim = NUM_ACTIONS - 1; current_claim >= 0; current_claim--) {
                // Visit claim nodes backward
                // "I have accepted claim 'current_claim', what is the value of the higher claims I could make?"
                if (current_claim < NUM_ACTIONS - 1) {
                    Node& node = claim_nodes[current_claim][roll_after_accepting_claim[current_claim]];
                    vector<double> action_probability = node.strategy;
                    node.utility = 0.0;

                    for (int a = 0; a < action_probability.size(); a++) {
                        int next_claim_idx = (current_claim == 0) ? (a + 1) : (current_claim + a + 1);

                        Node& next_node = response_nodes[next_claim_idx][roll_after_accepting_claim[next_claim_idx]];

                        double child_utility = -next_node.utility;
                        regret[a] = child_utility;
                        node.utility += action_probability[a] * child_utility;
                    }

                    // Accumulate regret for each possible claim
                    for (int a = 0; a < action_probability.size(); a++) {
                        regret[a] -= node.utility;
                        node.regret_sum[a] += node.p_opponent * regret[a];
                    }
                }

                // Visit response nodes backward
                // "Opponent just claimed 'current_claim', what is the value of Doubting or Continuing?"
                if (current_claim > 0) {
                    Node& node = response_nodes[current_claim][roll_after_accepting_claim[current_claim]];
                    vector<double> action_probability = node.strategy;
                    node.utility = 0.0;

                    // 0: Doubt
                    double doubt_utility =
                        get_dudo_utility(roll_after_accepting_claim[current_claim], current_claim - 1);
                    regret[0] = doubt_utility;
                    node.utility += action_probability[0] * doubt_utility;

                    // 1: Continue
                    if (action_probability.size() > 1) {
                        Node& next_node = claim_nodes[current_claim][roll_after_accepting_claim[current_claim]];
                        double accept_utility = next_node.utility;
                        regret[1] = accept_utility;
                        node.utility += action_probability[1] * accept_utility;
                    }

                    // Accumulate regret
                    for (int a = 0; a < action_probability.size(); a++) {
                        regret[a] -= node.utility;
                        node.regret_sum[a] += node.p_opponent * regret[a];
                    }
                }
            }

            // Reset strategy sums after half of training
            if (iteration == iterations / 2) {
                for (vector<Node> nodes : response_nodes)
                    for (Node& node : nodes)
                        for (int a = 0; a < node.strategy_sum.size(); a++) node.strategy_sum[a] = 0;
                for (vector<Node> nodes : claim_nodes)
                    for (Node& node : nodes)
                        for (int a = 0; a < node.strategy_sum.size(); a++) node.strategy_sum[a] = 0;
            }
        }
    }

    void save_strategies(const string& filename) {
        ofstream outfile(filename);

        if (!outfile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return;
        }

        outfile << "Initial Claim Policy:\n";
        outfile << left << setw(10) << "Roll" << "Probabilities (1x2 to 2x1)\n";
        outfile << string(80, '-') << "\n";

        for (int roll = 1; roll <= NUM_SIDES; roll++) {
            outfile << left << setw(10) << roll;

            const auto& vec = claim_nodes[0][roll].get_average_strategy();
            for (double p : vec) outfile << fixed << setprecision(2) << p << " ";

            outfile << "\n";
        }

        outfile << "\nResponse Policy (Dudo, Continue):\n";
        outfile << left << setw(20) << "Opponent Claim" << setw(10) << "Roll" << "Probabilities\n";
        outfile << string(80, '-') << "\n";

        for (int claim_idx = 1; claim_idx < NUM_ACTIONS; claim_idx++) {
            int array_idx = claim_idx - 1;
            string claim_str = to_string(CLAIM_NUM[array_idx]) + "x" + to_string(CLAIM_RANK[array_idx]);

            for (int roll = 1; roll <= NUM_SIDES; roll++) {
                outfile << left << setw(20) << claim_str << setw(10) << roll << "[";

                const auto& vec = response_nodes[claim_idx][roll].get_average_strategy();
                for (int i = 0; i < vec.size(); i++) {
                    outfile << fixed << setprecision(2) << vec[i];
                    if (i + 1 < vec.size()) outfile << ", ";
                }
                outfile << "]\n";
            }
        }

        outfile << "\nClaim Policy (Probabilities for all higher claims):\n";
        outfile << left << setw(20) << "Opponent Claim" << setw(10) << "Roll" << "Probabilities\n";
        outfile << string(80, '-') << "\n";

        for (int claim_idx = 1; claim_idx < NUM_ACTIONS - 1; claim_idx++) {
            int array_idx = claim_idx - 1;
            string claim_str = to_string(CLAIM_NUM[array_idx]) + "x" + to_string(CLAIM_RANK[array_idx]);

            for (int roll = 1; roll <= NUM_SIDES; roll++) {
                outfile << left << setw(20) << claim_str << setw(10) << roll << "[";

                const auto& vec = claim_nodes[claim_idx][roll].get_average_strategy();
                for (int i = 0; i < vec.size(); i++) {
                    outfile << fixed << setprecision(2) << vec[i];
                    if (i + 1 < vec.size()) outfile << ", ";
                }
                outfile << "]\n";
            }
        }

        outfile.close();
    }
};

int main() {
    DudoTrainer solver = DudoTrainer();
    solver.train(1'000'000);
    solver.save_strategies("strategies.txt");

    return 0;
}
