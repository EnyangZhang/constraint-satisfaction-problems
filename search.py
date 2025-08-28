# Inspired by AIFCA Python code Version 0.9.17
# Original source: https://aipython.org
#
# Artificial Intelligence: Foundations of Computational Agents
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
# https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# Adapted and extended by Anna Trofimova
# Professional Teaching Fellow, School of Computer Science
# University of Auckland
# 2025


from problem import Path
import heapq
import random
from csp import Constraint
from itertools import product

class FrontierPQ(object):
    """
    A frontier implemented as a priority queue (min-heap), storing (value, index, path) tuples, where:
    - value: the quantity to minimise (e.g., cost + heuristic)
    - index: a unique counter used to break ties (FIFO order)
    - path: a sequence of states representing the current path

    The priority queue returns the path with the smallest value first.
    """

    def __init__(self):
        """Constructs the frontier as an empty priority queue."""
        self.frontier_index = 0  # Unique counter to maintain insertion order
        self.frontierpq = []  # List to hold heap elements

    def add(self, path: Path, value: float) -> None:
        """
        Adds a path to the priority queue.

        Args:
            path: The path to be added.
            value: The priority value to minimise (e.g., cost or cost + heuristic).
        """
        self.frontier_index += 1
        # Use negative index to prioritise earlier-inserted paths in case of ties
        heapq.heappush(self.frontierpq, (value, -self.frontier_index, path))

    def pop(self) -> Path:
        """
        Removes and returns the path with the minimum value (highest priority).
        """
        _, _, path = heapq.heappop(self.frontierpq)
        return path

    def __repr__(self):
        """Returns a string representation of the frontier for debugging."""
        return str([(v, i, str(p)) for (v, i, p) in self.frontierpq])

    def __len__(self):
        """Returns the number of elements currently in the frontier."""
        return len(self.frontierpq)

    def __iter__(self):
        """Iterates over paths in the frontier (heap order, not sorted)."""
        for _, _, path in self.frontierpq:
            yield path


class Searcher:
    """
    A generic searcher for a graph search problem.

    Implements Depth-First Search (DFS) by default via a stack-based frontier.
    Subclasses override `initialize_frontier` and `add_to_frontier` for different strategies.
    """

    def __init__(self, problem):
        """
        Initializes the searcher with the given problem instance.

        Args:
            problem: An object representing the search problem.
        """
        self.problem = problem
        self.initialize_frontier()
        self.num_expanded = 0
        self.max_frontier_size = 0
        self.solution = None
        # Start search from the start node
        self.add_to_frontier(Path(problem.start_node()))

    def initialize_frontier(self):
        """Initializes the frontier as an empty stack (DFS default)."""
        self.frontier = []

    def is_empty(self):
        """Alternative check for frontier emptiness (used internally)."""
        return len(self.frontier) == 0

    def empty_frontier(self):
        """Checks whether the frontier is empty (external use)."""
        return len(self.frontier) == 0

    def add_to_frontier(self, path):
        """
        Adds a path to the frontier and updates the maximum frontier size.

        Args:
            path: A Path object representing the new path.
        """
        self.frontier.append(path)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))

    def search(self):
        """
        Performs a DFS-based search to find a goal.

        Returns:
            Path: A valid path to a goal state if found, else None.
        """
        while not self.empty_frontier():
            current_path = self.frontier.pop()  # DFS: LIFO
            self.num_expanded += 1

            if self.problem.is_goal(current_path.end()):
                self.solution = current_path
                return current_path

            # Expand and add neighbours to frontier
            for arc in reversed(list(self.problem.neighbors(current_path.end()))):
                new_path = Path(current_path, arc)
                self.add_to_frontier(new_path)

        return None  # No solution found


class BreadthFirstSearcher(Searcher):
    """
    Breadth-First Search (BFS) using a FIFO frontier.
    """

    def search(self):
        """
        Performs BFS until a goal is found or frontier is empty.

        Returns:
            Path: A valid path to a goal state if found, else None.
        """
        while not self.is_empty():
            current_path = self.frontier.pop(0)  # BFS: FIFO
            self.num_expanded += 1

            if self.problem.is_goal(current_path.end()):
                self.solution = current_path
                return current_path

            for arc in self.problem.neighbors(current_path.end()):
                new_path = Path(current_path, arc)
                self.add_to_frontier(new_path)

        return None


class AStarSearcher(Searcher):
    """
    A* Search using a priority queue frontier ordered by cost + heuristic.
    """

    def initialize_frontier(self):
        """Overrides default stack with priority queue."""
        self.frontier = FrontierPQ()

    def add_to_frontier(self, path: Path):
        """
        Adds path to the priority queue with f(n) = g(n) + h(n).
        """
        value = path.cost + self.problem.heuristic(path.end())
        self.frontier.add(path, value)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))


class UniformCostSearcher(Searcher):
    """
    Uniform Cost Search using a priority queue ordered by path cost g(n).
    """

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def add_to_frontier(self, path: Path):
        value = path.cost  # Priority is g(n)
        self.frontier.add(path, value)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))


class GreedySearcher(Searcher):
    """
    Greedy Best-First Search using a priority queue ordered by h(n).
    """

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def add_to_frontier(self, path: Path):
        value = self.problem.heuristic(path.end())  # Priority is h(n)
        self.frontier.add(path, value)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))


class IterativeDeepeningSearcher(Searcher):
    """
    Iterative Deepening Search (IDS): DFS with increasing depth limits.
    """

    def search_upto(self, depth_limit):
        """
        Performs depth-limited DFS.

        Args:
            depth_limit: The maximum depth to search.

        Returns:
            Path: A valid path to the goal state if found, else None.
        """
        while not self.empty_frontier():
            current_path = self.frontier.pop()
            self.num_expanded += 1

            if self.problem.is_goal(current_path.end()):
                self.solution = current_path
                return current_path

            if len(list(current_path.nodes())) >= depth_limit:
                continue

            for arc in reversed(list(self.problem.neighbors(current_path.end()))):
                new_path = Path(current_path, arc)
                self.add_to_frontier(new_path)

        return None

    def search(self):
        """
        Performs Iterative Deepening Search by gradually increasing the depth limit.

        Returns:
            Path: A valid path to the goal state if found, else None.
        """
        depth = 0

        path = None
        while not path:
            self.num_expanded = 0
            self.solution = None
            self.max_frontier_size = 0
            depth += 1
            path = self.search_upto(depth)

            # If not found, reset frontier before next iteration
            if not path:
                self.initialize_frontier()
                self.add_to_frontier(Path(self.problem.start_node()))

        return path


class HillClimbingSearcher:
    def __init__(self, csp, side_moves=0):
        """
        csp        : CSP instance
        side_moves : maximum allowed side moves (for side-move variant)
        """
        self.csp = csp
        self.side_moves = side_moves
        self.steps = 0

    def random_assignment(self):
        """Return a random assignment for all variables."""
        return {var: random.choice(var.domain) for var in self.csp.variables}

    def heuristic(self, assignment):
        """Number of violated constraints (lower is better)."""
        violations = 0
        for con in self.csp.constraints:
            if con.can_evaluate(assignment):
                if not con.holds(assignment):
                    violations += 1
        return violations

    def get_neighbors(self, assignment):
        """Generate all neighbors by changing one variable at a time."""
        neighbors = []
        for var in self.csp.variables:
            for value in var.domain:
                if value != assignment[var]:
                    new_assign = assignment.copy()
                    new_assign[var] = value
                    neighbors.append(new_assign)
        return neighbors

    def search(self):
        """Hill Climbing with optional side moves (no restarts)."""
        current = self.random_assignment()
        current_heur = self.heuristic(current)
        side_moves_left = self.side_moves

        while True:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break

            # Choose neighbors with heuristic <= current if side_moves, else < current
            if self.side_moves:
                candidates = [n for n in neighbors if self.heuristic(n) <= current_heur]
            else:
                candidates = [n for n in neighbors if self.heuristic(n) < current_heur]

            if not candidates:
                break

            next_state = random.choice(candidates)
            next_heur = self.heuristic(next_state)

            self.steps += 1

            if next_heur < current_heur:
                side_moves_left = self.side_moves  # reset on improvement
            elif next_heur == current_heur:
                side_moves_left -= 1
                if side_moves_left < 0:
                    break

            current = next_state
            current_heur = next_heur

        return current, current_heur


class VariableEliminationSolver(object):
    """Variable Elimination algorithm for CSPs."""

    def __init__(self, csp):
        """Initialize with a CSP instance."""
        self.csp = csp
        self.total_relation_size = 0
        self.max_relation_size = 0

    def run(self):
        """Run VE and return all consistent assignments."""
        vars_list = list(self.csp.variables)
        cons_list = list(self.csp.constraints)
        return self.eliminate(vars_list, cons_list)

    def eliminate(self, vars_set, constraints_set):
        """Recursive VE procedure."""
        if not vars_set:
            return [{}]  # empty assignment

        if len(vars_set) == 1:
            var = vars_set[0]
            assignments = []
            for val in var.domain:
                env = {var: val}
                if all(con.holds(env) for con in constraints_set if var in con.scope):
                    assignments.append(env)
            return assignments

        # pick variable to eliminate (simple: first one)
        X = vars_set[0]

        # constraints involving X
        CX = [con for con in constraints_set if X in con.scope]

        # join CX: all assignments over involved variables
        XR_vars = list({v for con in CX for v in con.scope})
        XR_domains = [v.domain for v in XR_vars]

        R = []
        for vals in product(*XR_domains):
            env = {v: val for v, val in zip(XR_vars, vals)}
            if all(con.holds(env) for con in CX):
                R.append(env)
        self.total_relation_size += len(R)
        self.max_relation_size = max(len(R), self.max_relation_size)
        # project R onto variables other than X
        other_vars = [v for v in XR_vars if v != X]
        NR = []
        for env in R:
            projected = {v: env[v] for v in other_vars}
            if projected not in NR:
                NR.append(projected)

        # remaining variables and constraints
        remaining_vars = [v for v in vars_set if v != X]
        remaining_constraints = [con for con in constraints_set if con not in CX]
        if other_vars:
            # create a new constraint for the projected relation
            proj_con = Constraint(
                other_vars,
                lambda *args, NR=NR, other_vars=other_vars: any(dict(zip(other_vars, args)) == r for r in NR),
                "proj"
            )
            remaining_constraints.append(proj_con)

        # recurse
        S = self.eliminate(remaining_vars, remaining_constraints)

        # join R and S
        final_result = []
        for r_env in R:
            for s_env in S:
                if all(r_env.get(v) == s_env.get(v) for v in r_env if v in s_env):
                    merged = {**r_env, **s_env}
                    final_result.append(merged)
        return final_result
