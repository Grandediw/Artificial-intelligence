#!/usr/bin/env python3

import math
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

# Time allowed per decision in seconds
TIME_LIMIT = 0.075
START_TIME = 0  # Global variable to track the start time of a move computation


class PlayerControllerMinimax(PlayerController):
    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the player agent.
        Continuously listens for game updates, computes the best move using Minimax, and sends it.
        """
        first_msg = self.receiver()  # Initial handshake message

        while True:
            msg = self.receiver()  # Receive game state update
            node = Node(message=msg, player=0)  # Parse the game state into a Node object
            best_move = self.search_best_next_move(initial_tree_node=node)  # Find the optimal move
            self.sender({"action": best_move, "search_time": None})  # Send the move back to the game engine

    def search_best_next_move(self, initial_tree_node):
        """
        Implements iterative deepening Minimax search with Alpha-Beta pruning.
        Continues to explore deeper levels until the time limit is exceeded.
        """
        global START_TIME

        START_TIME = time.time()  # Start timer for this move
        depth = 1  # Initial search depth
        best_move = 0  # Default move (e.g., STAY)
        max_score = -math.inf  # Start with the worst possible score for maximizing player

        # Caches for state evaluations and move ordering
        memory = dict()
        move_ordering = dict()

        # Perform iterative deepening until time runs out
        while (time.time() - START_TIME) <= TIME_LIMIT * 0.78:
            score, move = self.alphabeta(
                node=initial_tree_node,
                a=-math.inf,
                b=math.inf,
                depth=depth,
                memory=memory,
                move_ordering=move_ordering,
            )
            # Update the best move if a higher score is found
            if score > max_score:
                max_score = score
                best_move = move
            depth += 1  # Increase depth for the next iteration
        return ACTION_TO_STR[best_move]  # Convert the best move to a human-readable action string

    def alphabeta(self, node, a, b, depth, memory, move_ordering):
        """
        Alpha-Beta pruning implementation of the Minimax algorithm.
        Efficiently evaluates the game tree by pruning unnecessary branches.
        """
        state = node.state
        player = state.get_player()  # Current player (0 for green, 1 for red)
        hashable_state = HashableState(state)  # Convert the state into a hashable representation

        # Return cached evaluation if this state has already been computed at a sufficient depth
        if hashable_state in memory and memory[hashable_state][1] >= depth:
            return memory[hashable_state][0], node.move

        # Terminal condition: maximum depth reached, no fish left, or time is running out
        if (
            depth == 0
            or not state.get_fish_positions()
            or time.time() - START_TIME > 0.75 * TIME_LIMIT
        ):
            if node.move is None:
                # Return extreme scores for terminal nodes with no valid moves
                return -math.inf if player == 0 else math.inf, -1
            return self.h(state), node.move  # Evaluate the state with the heuristic function

        # Retrieve or compute children nodes
        children = move_ordering.get(hashable_state) or node.compute_and_get_children()
        children = [(child, i) for i, child in enumerate(children)]

        if player == 0:  # Maximizing player (Green Boat)
            v = -math.inf
            best_move = 0
            for child, _ in sorted(
                children,
                key=lambda x: move_ordering.get(hashable_state, {}).get(x[0], 0),
                reverse=True,
            ):
                score, _ = self.alphabeta(
                    node=child,
                    a=a,
                    b=b,
                    depth=depth - 1,
                    memory=memory,
                    move_ordering=move_ordering,
                )
                if score > v:
                    v = score
                    best_move = child.move
                a = max(a, v)
                if b <= a:  # Prune the branch
                    break
        else:  # Minimizing player (Red Boat)
            v = math.inf
            best_move = 0
            for child, _ in sorted(
                children,
                key=lambda x: move_ordering.get(hashable_state, {}).get(x[0], 0),
            ):
                score, _ = self.alphabeta(
                    node=child,
                    a=a,
                    b=b,
                    depth=depth - 1,
                    memory=memory,
                    move_ordering=move_ordering,
                )
                if score < v:
                    v = score
                    best_move = child.move
                b = min(b, v)
                if b <= a:  # Prune the branch
                    break

        # Cache the result for this state and depth
        memory[hashable_state] = (v, depth)
        move_ordering[hashable_state] = {
            child: self.h(child.state) for child, _ in children
        }

        return v, best_move

    def h(self, state):
        """
        Heuristic function to evaluate a game state.
        Considers the score difference and proximity to high-value fish.
        """
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        player_positions = state.get_hook_positions()

        # Start with the score difference
        score = state.player_scores[0] - state.player_scores[1]
        distance_and_score_h = 0

        # Calculate weighted contributions from the nearest fish
        fish_list = [
            (
                self.manhattan_dist(fish_positions[fish], player_positions[0]),
                fish_scores[fish],
            )
            for fish in fish_positions
            if fish_scores[fish] > 0
        ]
        fish_list.sort(key=lambda x: x[0])  # Sort fish by distance to the green hook

        for i in range(min(len(fish_list), 2)):  # Focus on the 2 nearest fish
            dist, fish_score = fish_list[i]
            distance_and_score_h += fish_score * (30 - dist) / 30

        return score + distance_and_score_h

    def manhattan_dist(self, f, p):
        """
        Compute the Manhattan distance on a toroidal grid.
        """
        return min(abs(p[0] - f[0]), 20 - abs(p[0] - f[0])) + abs(f[1] - p[1])


class HashableState:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        """
        Define equality for HashableState based on key game state attributes.
        """
        return (
            self.state.get_player() == other.state.get_player()
            and self.state.get_player_scores() == other.state.get_player_scores()
            and self.state.get_fish_positions() == other.state.get_fish_positions()
            and self.state.get_hook_positions() == other.state.get_hook_positions()
        )

    def __hash__(self):
        """
        Create a unique hash for the state based on its attributes.
        """
        return hash(
            (
                self.state.get_player(),
                tuple(self.state.get_player_scores()),
                tuple(sorted(self.state.get_fish_positions().items())),
                tuple(self.state.get_hook_positions().items()),
            )
        )
