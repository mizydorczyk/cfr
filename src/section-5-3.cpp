// Section 5.3
// Two-player Fixed-Strategy Iteration Counterfactual Regret Minimization (FSICFR) for Liar Dice.
// Adapted from "An Introduction to Counterfactual Regret Minimization" by Todd W. Neller and Marc Lanctot

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

const int DOUBT = 0, ACCEPT = 1;

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

    // Get Liar Die node current mixed strategy through regret-matching
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

    // Get Liar Die node average mixed strategy
    vector<double> get_average_strategy() {
        vector<double> avg(strategy_sum.size(), 0.0);
        double sum = accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0);

        for (int a = 0; a < strategy_sum.size(); a++) {
            avg[a] = (sum > 0) ? strategy_sum[a] / sum : 1.0 / strategy_sum.size();
        }

        return avg;
    }
};

class LiarDieTrainer {
   private:
    int sides;
    vector<vector<Node>> response_nodes;
    vector<vector<Node>> claim_nodes;
    mt19937 generator{0};

   public:
    LiarDieTrainer(int sides) : sides(sides) {
        response_nodes = vector<vector<Node>>(sides, vector<Node>(sides + 1));
        for (int my_claim = 0; my_claim <= sides; my_claim++) {
            for (int opponent_claim = my_claim + 1; opponent_claim <= sides; opponent_claim++) {
                response_nodes[my_claim][opponent_claim] =
                    Node((opponent_claim == 0 || opponent_claim == sides) ? 1 : 2);
            }
        }

        claim_nodes = vector<vector<Node>>(sides, vector<Node>(sides + 1));
        for (int opponent_claim = 0; opponent_claim < sides; opponent_claim++) {
            for (int roll = 1; roll <= sides; roll++) {
                claim_nodes[opponent_claim][roll] = Node(sides - opponent_claim);
            };
        }
    }

    void train(int iterations) {
        uniform_int_distribution<> dist(1, sides);
        vector<double> regret = vector<double>(sides);
        vector<int> roll_after_accepting_claim = vector<int>(sides);

        for (int iteration = 0; iteration < iterations; iteration++) {
            // Initialize rolls and starting probabilities
            for (int i = 0; i < roll_after_accepting_claim.size(); i++) {
                roll_after_accepting_claim[i] = dist(generator);
            }

            claim_nodes[0][roll_after_accepting_claim[0]].p_player = 1;
            claim_nodes[0][roll_after_accepting_claim[0]].p_opponent = 1;

            // Accumulate realization weights forward
            for (int opponent_claim = 0; opponent_claim <= sides; opponent_claim++) {
                // Visit response nodes forward
                if (opponent_claim > 0) {
                    for (int my_claim = 0; my_claim < opponent_claim; my_claim++) {
                        Node& node = response_nodes[my_claim][opponent_claim];
                        vector<double> action_probability = node.get_strategy();

                        if (opponent_claim < sides) {
                            Node& next_node = claim_nodes[opponent_claim][roll_after_accepting_claim[opponent_claim]];
                            next_node.p_player += action_probability[1] * node.p_player;
                            next_node.p_opponent += node.p_opponent;
                        }
                    }
                }

                // Visit claim nodes forward
                if (opponent_claim < sides) {
                    Node& node = claim_nodes[opponent_claim][roll_after_accepting_claim[opponent_claim]];
                    vector<double> action_probability = node.get_strategy();

                    for (int my_claim = opponent_claim + 1; my_claim <= sides; my_claim++) {
                        double next_claim_probability = action_probability[my_claim - opponent_claim - 1];
                        if (next_claim_probability > 0) {
                            Node& nextNode = response_nodes[opponent_claim][my_claim];
                            nextNode.p_player += node.p_opponent;
                            nextNode.p_opponent += next_claim_probability * node.p_player;
                        }
                    }
                }
            }

            // Backpropagate utilities, adjusting regrets and strategies
            for (int opponent_claim = sides; opponent_claim >= 0; opponent_claim--) {
                // Visit claim nodes backward
                if (opponent_claim < sides) {
                    Node& node = claim_nodes[opponent_claim][roll_after_accepting_claim[opponent_claim]];
                    vector<double> action_probability = node.strategy;
                    node.utility = 0.0;

                    for (int my_claim = opponent_claim + 1; my_claim <= sides; my_claim++) {
                        int action_index = my_claim - opponent_claim - 1;
                        Node& next_node = response_nodes[opponent_claim][my_claim];
                        double child_utility = -next_node.utility;
                        regret[action_index] = child_utility;
                        node.utility += action_probability[action_index] * child_utility;
                    }

                    for (int a = 0; a < action_probability.size(); a++) {
                        regret[a] -= node.utility;
                        node.regret_sum[a] += node.p_opponent * regret[a];
                    }

                    node.p_player = node.p_opponent = 0;
                }

                // Visit response nodes backward
                if (opponent_claim > 0) {
                    for (int my_claim = 0; my_claim < opponent_claim; my_claim++) {
                        Node& node = response_nodes[my_claim][opponent_claim];
                        vector<double> action_probability = node.strategy;
                        node.utility = 0.0;
                        double doubt_utility = (opponent_claim > roll_after_accepting_claim[my_claim]) ? 1 : -1;
                        regret[DOUBT] = doubt_utility;
                        node.utility += action_probability[DOUBT] * doubt_utility;

                        if (opponent_claim < sides) {
                            Node& next_node = claim_nodes[opponent_claim][roll_after_accepting_claim[opponent_claim]];
                            regret[ACCEPT] = next_node.utility;
                            node.utility += action_probability[ACCEPT] * next_node.utility;
                        }

                        for (int a = 0; a < action_probability.size(); a++) {
                            regret[a] -= node.utility;
                            node.regret_sum[a] += node.p_opponent * regret[a];
                        }

                        node.p_player = node.p_opponent = 0;
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
        outfile << left << setw(10) << "Roll" << "Probabilities\n";
        outfile << string(62, '-') << "\n";

        for (int roll = 1; roll <= sides; roll++) {
            outfile << left << setw(10) << roll;

            const auto& vec = claim_nodes[0][roll].get_average_strategy();
            for (double p : vec) outfile << fixed << setprecision(2) << p << " ";

            outfile << "\n";
        }

        outfile << "\nResponse Policy (Call or Continue):\n";
        outfile << left << setw(14) << "Old Claim" << setw(12) << "New Claim" << "Probabilities\n";
        outfile << string(62, '-') << "\n";

        for (int my_claim = 0; my_claim <= sides; my_claim++) {
            for (int opponent_claim = my_claim + 1; opponent_claim <= sides; opponent_claim++) {
                outfile << left << setw(14) << my_claim << setw(12) << opponent_claim << "[";

                const auto& vec = response_nodes[my_claim][opponent_claim].get_average_strategy();
                for (int i = 0; i < vec.size(); i++) {
                    outfile << fixed << setprecision(2) << vec[i];
                    if (i + 1 < vec.size()) outfile << ", ";
                }
                outfile << "]\n";
            }
        }

        outfile << "\nClaim Policy by Opponent Claim and Roll:\n";
        outfile << left << setw(18) << "Oppontent Claim" << setw(8) << "Roll" << "Probabilities\n";
        outfile << string(62, '-') << "\n";

        for (int opponent_claim = 0; opponent_claim < sides; opponent_claim++) {
            for (int roll = 1; roll <= sides; roll++) {
                outfile << left << setw(18) << opponent_claim << setw(8) << roll << "[";

                const auto& vec = claim_nodes[opponent_claim][roll].get_average_strategy();
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
    LiarDieTrainer solver = LiarDieTrainer(6);
    solver.train(1'000'000);
    solver.save_strategies("strategies.txt");

    return 0;
}
