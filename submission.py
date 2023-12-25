from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import math
import time


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # Get the current robot from the environment
    robot = env.get_robot(robot_id)
    # Get the opponent robot from the environment
    opponent = env.get_robot(not robot_id)

    # Extracting the robot's characteristics
    robot_position = robot.position
    robot_battery = robot.battery
    robot_credit = robot.credit
    robot_holding_package = robot.package

    # Extracting the opponent's characteristics
    opponent_position = opponent.position
    opponent_battery = opponent.battery
    opponent_credit = opponent.credit
    opponent_holding_package = opponent.package

    # Features weights when holding package:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    robot2destination_weight = -0.3

    # Features weights when not holding package:
    # ~~~~~~~~~~~~~~~~~~~~~
    robot2package_main_weight = 0.4  # main package - the best value package (the one that gives the best credit)
    robot2package_dist_average_weight = 0.3

    # Features weights global:
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    credit_weight = 11  # the weight of credit feature is higher than all other features
    battery_weight = 0.2
    opponent_battery_weight = -0.1

    # Features:
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    credit_feature = robot_credit

    package_main = None
    robot2package_main_dist = 0
    r2p_dist_sum = 0  # robot to packages distance sum
    heuristic = 0

    # If the robot is not holding a package:
    if not robot_holding_package:
        package_credit_to_battery_waste_max_ratio = 0
        battery_waste_main = 0

        packages_counter = 0

        # loop over all the available packages
        for package in [x for x in env.packages if x.on_board and not opponent.package == x]:
            # bug patch - sometimes the opponent is on the package destination, pick other package
            if opponent_battery == 0 and opponent_position == package.destination:
                continue
            r2p = manhattan_distance(robot_position, package.position)
            p2d = manhattan_distance(package.position, package.destination)
            # don't choose unreachable packages or robot in on package destination
            if robot_battery < r2p + p2d or r2p + p2d == 0:
                continue
            r2p_dist_sum += r2p

            package_credit_to_battery_waste_ratio = -r2p + 10 * p2d * 2 / (p2d + r2p)

            packages_counter += 1

            # choose the best value to battery waste package
            if package_credit_to_battery_waste_max_ratio < package_credit_to_battery_waste_ratio:
                package_credit_to_battery_waste_max_ratio = package_credit_to_battery_waste_ratio
                package_main = package
                robot2package_main_dist = r2p
                battery_waste_main = r2p + p2d
            elif package_credit_to_battery_waste_max_ratio == package_credit_to_battery_waste_ratio:
                if r2p + p2d < battery_waste_main:
                    package_credit_to_battery_waste_max_ratio = package_credit_to_battery_waste_ratio
                    package_main = package
                    robot2package_main_dist = r2p
                    battery_waste_main = r2p + p2d

    # Basic heuristic calculation:
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    robot_is_winning = robot_credit > opponent_credit and robot_battery >= opponent_battery

    if robot_is_winning:
        heuristic += 100000     # Bonus if winning state
        heuristic += (robot_credit - opponent_credit) * credit_weight   # Gets x2 points but losing points if opponent gains
    if robot_credit > opponent_credit:
        heuristic += 10000      # Bonus if have more credit state
    if robot_battery > opponent_battery:
        heuristic += 1000       # Bonus if have more battery state
    if robot_holding_package:
        # if the robot is holding a package, the heuristic decreases as the robot is far from destination
        robot2destination_feature = manhattan_distance(robot_position, robot.package.destination)
        # when holding a package, heuristic is bigger than if not holding a package,
        heuristic += 10         # Bonus if holding a package
        # heuristic decreases as the robot with a package is far from the destination
        heuristic += robot2destination_feature * robot2destination_weight
    if package_main and not robot_holding_package:
        robot2package_main_criteria = -robot2package_main_dist
        # get closer to all packages available on board
        robot2package_dist_average_criteria = -r2p_dist_sum / 2  # packages_counter
        heuristic += robot2package_main_criteria * robot2package_main_weight
        heuristic += robot2package_dist_average_criteria * robot2package_dist_average_weight

    heuristic += credit_feature * credit_weight + robot_battery * battery_weight + opponent_battery * opponent_battery_weight
    return heuristic


def eval_heuristic(env: WarehouseEnv, agent_id: int):
    # Utility - if Pg(s)
    if env.done():
        balances = env.get_balances()
        if balances[agent_id] == balances[(agent_id + 1) % 2]:
            return 0
        elif balances[agent_id] > balances[(agent_id + 1) % 2]:
            return 2000000
        else:
            return -2000000
    # Heuristic - else
    else:
        heuristic = smart_heuristic(env, agent_id)
        return heuristic


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self):
        self.start_time = 0
        # self.time_limit = 1
        self.max_diffs = 0
        self.min_diffs = 0
        self.epsilon = 0.0001  # sec
        self.safety_time = 0.1
        self.depth_explored = 0
        self.debug = True  # To print additional data like time spent or depth explored etc..

    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.depth_explored = 0
        self.max_diffs = 0
        self.min_diffs = 0
        self.start_time = time.time()
        # self.time_limit = time_limit - self.epsilon
        best_action = self.max_value_first_step(env, agent_id, time_limit - self.safety_time)
        if self.debug:
            print("depth explored minimax = " + str(self.depth_explored))
            print("best action = " + best_action)
        return best_action


    def max_value_first_step(self, env: WarehouseEnv, agent_id, time_left: float) -> float:
        curMax = -math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        op_chosen = operators[0]
        if operators.__contains__("drop off") or operators.__contains__("pick up"):
            operators = operators[::-1]
            children = children[::-1]

        time_spent_sum = 0
        if self.debug:
            print("Operators: " + str(operators))
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)

            time_spent = time.time() - self.start_time
            step_time = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            time_before_operation = time.time()
            v, min_depth, max_depth = self.min_value(child, (agent_id + 1) % 2, step_time, 1)

            if v > curMax:
                curMax = max(v, curMax)
                op_chosen = op
            if self.debug:
                print("Operator: " + str(op) + " Min: " + str(min_depth) + " Max: " + str(max_depth)
                      + " heur: " + "{:.5f}".format(v) + " time: " + "{:.2f}".format(time.time() - time_before_operation))
                time_spent_sum += time.time() - time_before_operation
        if self.debug:
            print("timeSpent: " + str(time_spent_sum))
            print("diffs: " + str(self.max_diffs))
        return op_chosen

    def max_value(self, env: WarehouseEnv, agent_id: int, time_left: float, depth: int) -> (float, int, int):
        start_time = time.time()
        if env.done() or time_left <= self.epsilon:
            return eval_heuristic(env, agent_id), 0, 0
        min_depth = math.inf
        max_depth = 0
        curMax = -math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)

            time_spent = time.time() - start_time
            time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            v, depth1, depth2 = self.min_value(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)
            curMax = max(v, curMax)

            if self.debug:
                min_depth = min(min_depth, depth1)
                max_depth = max(max_depth, depth2)
                self.max_diffs += max_depth - depth2

        if self.debug:
            self.depth_explored += 1
        return curMax, min_depth + 1, max_depth + 1

    def min_value(self, env: WarehouseEnv, agent_id: int, time_left: float, depth: int) -> (float, int, int):
        start_time = time.time()
        if env.done() or time_left <= self.epsilon:
            return eval_heuristic(env, (agent_id + 1) % 2), 0, 0
        min_depth = math.inf
        max_depth = 0
        curMin = math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)

            time_spent = time.time() - start_time
            time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            v, depth1, depth2 = self.max_value(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)

            curMin = min(v, curMin)
            if self.debug:
                min_depth = min(min_depth, depth1)
                max_depth = max(max_depth, depth2)
                self.max_diffs += max_depth - depth2

        if self.debug:
            self.depth_explored += 1
        return curMin, min_depth + 1, max_depth + 1


class AgentAlphaBeta(Agent):
    def __init__(self):
        self.start_time = 0
        self.diffs = 0
        # self.time_limit = 1
        self.epsilon = 0.0001  # sec
        self.time_safety = 0.1
        self.depth_explored = 0
        self.debug = True      # To print additional data like time spent or depth explored etc..

    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.depth_explored = 0
        self.start_time = time.time()
        # self.time_limit = time_limit - self.epsilon
        best_action = self.max_value_first_step(env, agent_id, time_limit - self.time_safety)
        if self.debug:
            print("depth explored alpha beta = " + str(self.depth_explored))
            print("best action = " + best_action)
        return best_action

    def max_value_first_step(self, env: WarehouseEnv, agent_id, time_left) -> float:
        curMax = -math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        op_chosen = operators[0]
        if operators.__contains__("drop off") or operators.__contains__("pick up"):
            operators = operators[::-1]
            children = children[::-1]
        time_spent_sum = 0
        if self.debug:
            print("Operators: " + str(operators))
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)

            time_spent = time.time() - self.start_time
            step_time = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            time_before_operation = time.time()

            v, min_depth, max_depth = self.min_value(child, (agent_id + 1) % 2, step_time, 1, -math.inf, math.inf)
            if self.debug:
                print("Operator: " + str(op) + " Min: " + str(min_depth) + " Max: " + str(max_depth)
                      + " time: " + "{:.2f}".format(time.time() - time_before_operation) + " heur: " + "{:.5f}".format(v))
                time_spent_sum += time.time() - time_before_operation

            if v > curMax:
                curMax = max(v, curMax)
                op_chosen = op
        if self.debug:
            print(" diffs: " + str(self.diffs))
            print("timeSpent: " + str(time_spent_sum))
        return op_chosen

    def max_value(self, env: WarehouseEnv, agent_id, time_left, depth, alpha, beta) -> (float, int, int):
        start_time = time.time()
        if env.done() or time_left <= 0:
            return eval_heuristic(env, agent_id), 0, 0
        min_depth = math.inf
        max_depth = 0
        curMax = -math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            time_spent = time.time() - start_time
            time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            v, depth1, depth2 = self.min_value(child, (agent_id + 1) % 2, time_left_per_step, depth + 1, alpha, beta)
            curMax = max(v, curMax)
            alpha = max(curMax, alpha)
            if self.debug:
                min_depth = min(min_depth, depth1)
                max_depth = max(max_depth, depth2)
            if curMax >= beta:
                return math.inf, min_depth + 1, max_depth + 1
        if self.debug:
            self.diffs += max_depth - min_depth
            self.depth_explored += 1
        return curMax, min_depth + 1, max_depth + 1

    def min_value(self, env: WarehouseEnv, agent_id, time_left, depth, alpha, beta) -> (float, int, int):
        start_time = time.time()
        if env.done() or time_left <= 0:
            return eval_heuristic(env, (agent_id + 1) % 2), 0, 0
        min_depth = math.inf
        max_depth = 0
        curMin = math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            time_spent = time.time() - start_time
            time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            v, depth1, depth2 = self.max_value(child, (agent_id + 1) % 2, time_left_per_step, depth + 1, alpha, beta)
            curMin = min(v, curMin)
            beta = min(curMin, beta)
            if self.debug:
                min_depth = min(min_depth, depth1)
                max_depth = max(max_depth, depth2)
            if curMin <= alpha:
                return -math.inf, min_depth + 1, max_depth + 1
        if self.debug:
            self.diffs += max_depth - min_depth
            self.depth_explored += 1
        return curMin, min_depth + 1, max_depth + 1


class AgentExpectimax_firstImplementation(Agent):
        def __init__(self):
            self.start_time = 0
            self.time_limit = 1
            self.epsilon = 0.0001  # sec
            self.time_safety = 0.1
            self.depth_explored = 0
            self.player_id = 0
            self.debug = False      # To print additional data like time spent or depth explored etc..

        # TODO: section d : 1
        def run_step(self, env: WarehouseEnv, agent_id, time_limit):
            self.player_id = agent_id
            self.depth_explored = 0
            self.start_time = time.time()
            self.time_limit = time_limit - self.epsilon
            best_action = self.max_value_first_step(env, agent_id, time_limit - self.time_safety)

            if self.debug:
                print("depth explored expectimax = " + str(self.depth_explored))
                print("best action = " + best_action)

            return best_action

        def max_value_first_step(self, env: WarehouseEnv, agent_id, time_left) -> float:
            curMax = -math.inf
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            steps_left = len(children)
            op_chosen = operators[0]
            if operators.__contains__("drop off") or operators.__contains__("pick up"):
                operators = operators[::-1]
                children = children[::-1]
            if self.debug:
                print("Operators: " + str(operators))
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)

                time_spent = time.time() - self.start_time
                step_time = (time_left - time_spent) / steps_left - self.epsilon
                steps_left -= 1
                time_before_operation = time.time()
                v = self.RB_expectimax(child, (agent_id + 1) % 2, step_time, 1)

                if self.debug:
                    print("Operator: " + str(op) + " heur: " + "{:.6f}".format(v) + " time: " + str(time.time() - time_before_operation))
                if v > curMax:
                    curMax = max(v, curMax)
                    op_chosen = op
            return op_chosen

        def calc_prob(self, env: WarehouseEnv, agent_id) -> float:
            operators = env.get_legal_operators(agent_id)
            charges_count = 0
            if not operators.__contains__("park"):
                charges_count = len([x for x in env.charge_stations if manhattan_distance(env.get_robot(agent_id).position, x.position) == 1])
            return 1 / (len(operators) + charges_count)

        def RB_expectimax(self, env: WarehouseEnv, agent_id, time_left, depth) -> float:
            start_time = time.time()
            if env.done() or time_left <= 0:
                return eval_heuristic(env, self.player_id)
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            steps_left = len(children)
            # chance nodes:
            if agent_id != self.player_id:
                sum_v = 0
                prob = self.calc_prob(env, agent_id)
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                    time_spent = time.time() - start_time
                    time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
                    steps_left -= 1
                    v = prob*self.RB_expectimax(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)
                    # if the opponent moved to a charging station,
                    # it will have zero distance from charging station
                    # giving 2 times the probability for the operator "moving to charging station"
                    if len([x for x in child.charge_stations if (manhattan_distance(child.get_robot(agent_id).position, x.position) == 0)]) > 0 and prob < 1:
                        v = 2*v
                    sum_v += v
                return sum_v
            # max nodes:
            else:
                curMax = -math.inf
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                    time_spent = time.time() - start_time
                    time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
                    steps_left -= 1
                    v = self.RB_expectimax(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)
                    curMax = max(v, curMax)
                return curMax


class AgentExpectimax(Agent):
    def __init__(self):
        self.start_time = 0
        self.time_limit = 1
        self.epsilon = 0.0001  # sec
        self.time_safety = 0.1
        self.depth_explored = 0
        self.player_id = 0
        self.debug = True  # To print additional data like time spent or depth explored etc..

    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.player_id = agent_id
        self.depth_explored = 0
        self.start_time = time.time()
        self.time_limit = time_limit - self.epsilon
        best_action = self.max_value_first_step(env, agent_id, time_limit - self.time_safety)

        if self.debug:
            print("depth explored expectimax = " + str(self.depth_explored))
            print("best action = " + best_action)

        return best_action

    def max_value_first_step(self, env: WarehouseEnv, agent_id, time_left) -> float:
        curMax = -math.inf
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        op_chosen = operators[0]
        if operators.__contains__("drop off") or operators.__contains__("pick up"):
            operators = operators[::-1]
            children = children[::-1]
        if self.debug:
            print("Operators: " + str(operators))
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)

            time_spent = time.time() - self.start_time
            step_time = (time_left - time_spent) / steps_left - self.epsilon
            steps_left -= 1
            time_before_operation = time.time()
            v = self.RB_expectimax(child, (agent_id + 1) % 2, step_time, 1)

            if self.debug:
                print("Operator: " + str(op) + " heur: " + "{:.6f}".format(v) + " time: " + str(
                    time.time() - time_before_operation))
            if v > curMax:
                curMax = max(v, curMax)
                op_chosen = op
        return op_chosen

    def calc_prob(self, env: WarehouseEnv, agent_id) -> float:
        operators = env.get_legal_operators(agent_id)
        charges_count = 0
        if not operators.__contains__("park"):
            charges_count = len([x for x in env.charge_stations if
                                 manhattan_distance(env.get_robot(agent_id).position, x.position) == 1])
        return 1 / (len(operators) + charges_count)

    def RB_expectimax(self, env: WarehouseEnv, agent_id, time_left, depth) -> float:
        start_time = time.time()
        if env.done() or time_left <= 0:
            return eval_heuristic(env, self.player_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        steps_left = len(children)
        if agent_id != self.player_id:
            # chance nodes:
            if len([x for x in env.charge_stations if
                                 manhattan_distance(env.get_robot(agent_id).position, x.position) == 1]) > 1:
                sum_v = 0
                prob = self.calc_prob(env, agent_id)
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                    time_spent = time.time() - start_time
                    time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
                    steps_left -= 1
                    v = prob * self.RB_expectimax(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)
                    # if the opponent moved to a charging station,
                    # it will have zero distance from charging station
                    # giving 2 times the probability for the operator "moving to charging station"
                    if len([x for x in child.charge_stations if
                            (manhattan_distance(child.get_robot(agent_id).position,
                                                x.position) == 0)]) > 0 and prob < 1:
                        v = 2 * v
                    sum_v += v
                return sum_v
            # min nodes:
            else:
                curMin = math.inf
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)
                    time_spent = time.time() - start_time
                    time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
                    steps_left -= 1
                    v = self.RB_expectimax(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)
                    curMin = min(v, curMin)
                return curMin
        # max nodes:
        else:
            curMax = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                time_spent = time.time() - start_time
                time_left_per_step = (time_left - time_spent) / steps_left - self.epsilon
                steps_left -= 1
                v = self.RB_expectimax(child, (agent_id + 1) % 2, time_left_per_step, depth + 1)
                curMax = max(v, curMax)
            return curMax


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)


class AgentHardCoded_TestExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        position = env.get_robot(robot_id).position

        if ("move north" in env.get_legal_operators(robot_id)) and len([x for x in env.charge_stations if manhattan_distance(tuple(a + b for a, b in zip(position, (0, -1))), x.position) == 0]) > 0:
            return 'move north'
        if ("move south" in env.get_legal_operators(robot_id)) and len([x for x in env.charge_stations if manhattan_distance(tuple(a + b for a, b in zip(position, (0, 1))), x.position) == 0]) > 0:
            return 'move south'
        if ("move east" in env.get_legal_operators(robot_id)) and len([x for x in env.charge_stations if manhattan_distance(tuple(a + b for a, b in zip(position, (1, 0))), x.position) == 0]) > 0:
            return 'move east'
        if ("move west" in env.get_legal_operators(robot_id)) and len([x for x in env.charge_stations if manhattan_distance(tuple(a + b for a, b in zip(position, (-1, 0))), x.position) == 0]) > 0:
            return 'move west'

        return self.run_random_step(env, robot_id, time_limit)

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
