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

from problem import *



class Variable(object):
    """A variable.
    name (string) - name of the variable
    domain (list) - a list of the values for the variable.
    an (x,y) position for displaying
    """

    def __init__(self, name, domain, position=None):
        """Variable
        name a string
        domain a list of printable values
        position of form (x,y) where 0 <= x <= 1, 0 <= y <= 1
        """
        self.name = name  # string
        self.domain = domain  # list of values
        self.position = position if position else (random.random(), random.random())
        self.size = len(domain)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name  # f"Variable({self.name})"


class Constraint(object):
    """A Constraint consists of
    * scope: a tuple or list of variables
    * condition: a Boolean function that can applied to a tuple of values for variables in scope
    * string: a string for printing the constraint
    """

    def __init__(self, scope, condition, string=None, position=None):
        self.scope = scope
        self.condition = condition
        self.string = string
        self.position = position

    def __repr__(self):
        return self.string

    def can_evaluate(self, assignment):
        """
        assignment is a variable:value dictionary
        returns True if the constraint can be evaluated given assignment
        """
        return all(v in assignment for v in self.scope)

    def holds(self, assignment):
        """returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment, ie self.can_evaluate(assignment) is true
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))


class CSP(object):
    """A CSP consists of
    * a title (a string)
    * variables, a list or set of variables
    * constraints, a list of constraints
    * var_to_const, a variable to set of constraints dictionary
    """

    def __init__(self, title, variables, constraints):
        """title is a string
        variables is set of variables
        constraints is a list of constraints
        """
        self.title = title
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var: set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """string representation of CSP"""
        return self.title

    def __repr__(self):
        """more detailed string representation of CSP"""
        return f"CSP({self.title}, {self.variables}, {([str(c) for c in self.constraints])})"

    def consistent(self, assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(con.holds(assignment)
                   for con in self.constraints
                   if con.can_evaluate(assignment))

    def is_goal(self, node):
        """returns whether the current node is a goal for the search
        """
        return len(node) == len(self.variables)

    def start_node(self):
        """returns the start node for the search
        """
        return {}

    def neighbors(self, node):
        """returns a list of the neighboring nodes of node.
        """
        var = self.variables[len(node)]  # the next variable
        res = []
        for val in var.domain:
            new_env = node | {var: val}  # dictionary union
            if self.consistent(new_env):
                res.append(Arc(node, new_env))
        return res

    def show_constraint_graph(self, with_labels=True):
        """Visualise the CSP as a constraint graph using networkx,
        using Variable.position for layout if available.
        """
        G = nx.Graph()

        # add nodes with positions
        pos = {}
        for var in self.variables:
            G.add_node(var.name)
            # scale positions so they work nicely with matplotlib
            pos[var.name] = (var.position[0], var.position[1])

        # add edges: connect variables that share a constraint
        for con in self.constraints:
            scope_vars = [var.name for var in con.scope]
            if len(scope_vars) == 2:
                G.add_edge(scope_vars[0], scope_vars[1], label=con.string)
            elif len(scope_vars) > 2:
                # hyper-constraint: connect all vars pairwise
                for i in range(len(scope_vars)):
                    for j in range(i + 1, len(scope_vars)):
                        G.add_edge(scope_vars[i], scope_vars[j], label=con.string)

        # draw nodes/edges
        nx.draw(G, pos, with_labels=with_labels, node_size=1200,
                node_color="lightblue", font_size=10, font_weight="bold")

        # optional: draw constraint labels on edges
        edge_labels = nx.get_edge_attributes(G, 'label')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                         font_color='red', font_size=7)

        plt.title(f"Constraint Graph for {self.title}")
        plt.show()




