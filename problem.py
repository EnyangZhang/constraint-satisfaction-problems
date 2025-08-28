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

import random
import networkx as nx
import matplotlib.pyplot as plt
import random
from copy import deepcopy

class GraphSearchProblem:
    """
    Represents a search problem defined on an explicit graph.

    The graph consists of:
    - A set of nodes.
    - A set of arcs (directed edges with costs).
    - A start node and one or more goal nodes.
    - An optional heuristic map and node positions for visualisation.
    """

    def __init__(self, nodes, arcs, start=None, goals=set(), hmap={}, positions=None):
        """
        Initializes the search problem with nodes, arcs, and optional parameters.

        Parameters:
        - nodes: Iterable of nodes in the graph.
        - arcs: Iterable of Arc objects (directed edges with cost).
        - start: The start node (optional).
        - goals: A set of goal nodes (default: empty set).
        - hmap: A dictionary mapping nodes to heuristic values (default: {}).
        - positions: Optional dictionary mapping nodes to 2D coordinates.
        """
        self.nodes = nodes
        self.arcs = arcs
        self.start = start
        self.goals = goals
        self.hmap = hmap

        # Build a mapping from each node to its outgoing arcs (neighbours)
        self.neighbours = {node: [] for node in nodes}
        for arc in arcs:
            self.neighbours[arc.from_node].append(arc)

        # Assign positions for drawing (use provided or random)
        self.positions = positions if positions is not None else {
            node: (random.random(), random.random()) for node in nodes
        }

    def start_node(self):
        """Returns the start node of the search problem."""
        return self.start

    def is_goal(self, node):
        """Returns True if the given node is a goal node."""
        return node in self.goals

    def neighbors(self, node):
        """
        Returns a list of outgoing arcs from the given node.

        Each arc leads to a neighbouring node.
        """
        return self.neighbours[node]

    def heuristic(self, node):
        """
        Returns the heuristic value of the given node.

        Returns 0 if the node is not in the heuristic map.
        """
        return self.hmap.get(node, 0)

    def __repr__(self):
        """Returns a string representation of all arcs in the graph."""
        return "  ".join(str(arc) for arc in self.arcs)

    def show(self, show_costs=True, fontsize=10, show_heuristics=True):
        """
        Visualises the graph using NetworkX and Matplotlib.

        Parameters:
        - show_costs: If True, displays the cost of each arc.
        - fontsize: Font size for labels.
        - show_heuristics: If True, shows heuristic values inside node labels.
        """
        G = nx.DiGraph()

        # Add edges with weight (used for displaying arc cost)
        for arc in self.arcs:
            G.add_edge(arc.from_node, arc.to_node, weight=arc.cost)

        # Colour nodes based on their role (start, goal, or other)
        node_colors = []
        for node in G.nodes():
            if node == self.start:
                node_colors.append('green')
            elif node in self.goals:
                node_colors.append('red')
            else:
                node_colors.append('lightgrey')

        # Create node labels (with or without heuristics)
        if show_heuristics:
            labels = {node: f"{node}\nh={self.hmap.get(node, 0)}" for node in G.nodes()}
        else:
            labels = {node: node for node in G.nodes()}

        # Set up the plot
        plt.figure(figsize=(10, 8))

        # Draw the graph: nodes and edges
        nx.draw(
            G,
            pos=self.positions,
            with_labels=False,
            arrows=True,
            node_color=node_colors,
            edge_color='black',
            node_size=2000,
            linewidths=1.5
        )

        # Draw labels inside nodes
        nx.draw_networkx_labels(
            G,
            pos=self.positions,
            labels=labels,
            font_size=fontsize,
            font_color='black'
        )

        # Draw arc cost labels on the edges
        if show_costs:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(
                G,
                pos=self.positions,
                edge_labels=edge_labels,
                font_size=fontsize
            )

        plt.axis('off')
        plt.show()


class Path:
    """
    A Path represents a sequence of arcs connecting nodes in the graph.

    A path is recursively defined:
    - A single node (base case).
    - A previous Path extended by an Arc (recursive case).
    """

    def __init__(self, initial, arc=None):
        """
        Create a new path.

        Parameters:
        - initial: Either a starting node (if arc is None), or a Path.
        - arc: An Arc used to extend the initial path.
        """
        self.initial = initial
        self.arc = arc

        # Compute path cost
        if arc is None:
            self.cost = 0
        else:
            self.cost = initial.cost + arc.cost

    def end(self):
        """
        Returns the last node of the path (i.e., the current state).
        """
        return self.initial if self.arc is None else self.arc.to_node

    def nodes(self):
        """
        Yields the sequence of nodes in the path, from end to start.

        Use list(reversed(list(p.nodes()))) to get start-to-end order.
        """
        current = self
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial
        yield current.initial

    def initial_nodes(self):
        """
        Yields all nodes in the path *excluding* the last (end) node.

        Useful for cycle detection or tracing the path prefix.
        """
        if self.arc is not None:
            yield from self.initial.nodes()

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return (list(reversed(list(self.nodes()))) ==
                list(reversed(list(other.nodes()))) and
                self.cost == other.cost)

    def __repr__(self):
        """
        Returns a string representation of the path.

        Shows transitions between nodes. If actions exist, they can be added here.
        """
        if self.arc is None:
            return str(self.initial)
        else:
            return f"{self.initial} --> {self.arc.to_node}"


class Arc:
    """
    An Arc represents a directed edge between two nodes, with an optional cost and action.
    """

    def __init__(self, from_node, to_node, cost=1):
        """
        Initializes the arc with a source, target, and non-negative cost.

        Parameters:
        - from_node: Source node.
        - to_node: Target node.
        - cost: Numeric cost (default 1). Must be non-negative.
        """
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost
        assert cost >= 0, (f"Cost cannot be negative: {self}, cost={cost}")

    def __repr__(self):
        """Returns a string representation of the arc."""
        return f"{self.from_node} --> {self.to_node}"


