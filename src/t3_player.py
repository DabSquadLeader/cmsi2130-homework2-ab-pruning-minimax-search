"""
Artificial Intelligence responsible for playing the game of T3!
Implements the alpha-beta-pruning mini-max search algorithm
"""
from dataclasses import *
from typing import *
from t3_state import *
    
def choose(state: "T3State") -> Optional["T3Action"]:
    """
    Main workhorse of the T3Player that makes the optimal decision from the max node
    state given by the parameter to play the game of Tic-Tac-Total.
    
    [!] Remember the tie-breaking criteria! Moves should be selected in order of:
    1. Best utility
    2. Smallest depth of terminal
    3. Earliest move (i.e., lowest col, then row, then move number)
    
    You can view tiebreaking as something of an if-ladder: i.e., only continue to
    evaluate the depth if two candidates have the same utility, only continue to
    evaluate the earliest move if two candidates have the same utility and depth.
    
    Parameters:
        state (T3State):
            The board state from which the agent is making a choice. The board
            state will be either the odds or evens player's turn, and the agent
            should use the T3State methods to simplify its logic to work in
            either case.
    
    Returns:
        Optional[T3Action]:
            If the given state is a terminal (i.e., a win or tie), returns None.
            Otherwise, returns the best T3Action the current player could take
            from the given state by the criteria stated above.
            
    """
    # Set default values for a and B, start with a max node
    a = float('-inf')
    B = float('inf')
    is_max = True
    is_min = False
    # Uses alpha-beta pruning on a minimax tree to find the most optimal solution to the game
    best_score, optimal_action, _ = alphabeta(state, a, B, is_max, is_min, 0)
    return optimal_action

def alphabeta(state: Any, a: float, B: float, is_max: bool, is_min: bool, node_depth: int) -> Tuple[float, None, int]:
    """
    Implements the alpha-beta pruning minimax algorithm to evaluate the utility of a given game state and choose
    the optimal move.

    Parameters:
        state (Any): 
            The current game state that the agent is making decisions from.
        a (float): 
            Variable (alpha) representing the highest utility value of the node.
        B (float): 
            Variable (beta) representing the lowest utility value of the node.
        is_max (bool): 
            Indicates that the node is a max node.
        is_min (bool): 
            Indicates that the node is a min node.
        node_depth (int): 
            The current depth of the game tree node.
    
    Returns:
        Tuple[float, None, int]: 
            A tuple that includes the best utility score for the current player and the depth of the terminal node.
    """
    # Checks code if the state is a win, tie, or loss and returns relevant values for each
    check_win = state.is_win()
    check_tie = state.is_tie()
    
    if check_tie:
        return (0.5, None, node_depth)
    if check_win:
        # Only returns a win if a min node
        if not is_max and is_min:
            return (1, None, node_depth)
        else:
            return (0, None, node_depth)
    
    # Generates possible moves for the node and updates starting variables
    moves = state.get_transitions()
    optimal_move = None
    lowest_depth = node_depth
    
    # Takes max node and creates children (if any) to find best action, score and depth
    if is_max and not is_min:
        best_score = float('-inf') # Sets lowest possible score to -infinity by default
        for action, child in moves:
            # Creates min node from parent max node
            is_max = False
            is_min = True
            utility_score, _, new_depth = alphabeta(child, a, B, is_max, is_min, node_depth + 1)
            
            # Determines best score, optimal move, and lowest depth of the child
            best_score, optimal_move, lowest_depth = best_choice_max(utility_score, best_score,
                                                                    action, optimal_move,
                                                                    new_depth, lowest_depth)
            
            # Updates a if the best score is more than the current a value
            a = max(a, best_score)
            # Prune if B is less than/equal to a
            if B <= a:
                break
        return best_score, optimal_move, lowest_depth

    # Takes min node and creates children (if any) to find best action, score and depth
    elif is_min and not is_max:
        best_score = float('inf') # Sets highest possible score to infinity by default
        for action, child in moves:
            # Creates min node from parent max node
            is_min = False
            is_max = True
            utility_score, _, new_depth = alphabeta(child, a, B, is_max, is_min, node_depth + 1)
            
            # Determines best score, optimal move, and lowest depth of the child
            best_score, optimal_move, lowest_depth = best_choice_min(utility_score, best_score,
                                                                    action, optimal_move,
                                                                    new_depth, lowest_depth)
            # Updates B if the best score is less than the current B value
            B = min(B, best_score)
            # Prune if B is less than/equal to a
            if B <= a:
                break
        return best_score, optimal_move, lowest_depth
    
    return (0, None, node_depth)  

def best_choice_max(utility_score: float, best_score: float, action: Any, optimal_move: Any, 
                    new_depth: int, lowest_depth: int) -> Tuple[float, Any, int]:
    """
    Helper function to select the best action for the max node.
    Updates the utility score, the best action, and the lowest depth based on superior values of each.

    Parameters:
        utility_score (float): 
            The utility score of the current action.
        best_score (float): 
            The best utility score found so far.
        action (Any): 
            The current action being evaluated.
        optimal_move (Any): 
            The most optimal action found so far.
        new_depth (int): 
            The depth of the node of the current action.
        lowest_depth (int): 
            The lowest depth of a terminal node found so far.
    
    Returns:
        Tuple[float, Any, int]: 
            A tuple containing the updated best utility score, the optimal move, and the lowest depth of the terminal node.
    """
    if utility_score > best_score: # Max node wants the highest possible score
        return utility_score, action, new_depth
    elif utility_score == best_score: # Tiebreaker to choose node w/ lower depth
        if new_depth < lowest_depth:
            return utility_score, action, new_depth
        elif new_depth == lowest_depth and action < optimal_move: # Tiebreaker to choose node w/ higher move
            return best_score, action, lowest_depth
    return best_score, optimal_move, lowest_depth

def best_choice_min(utility_score: float, best_score: float, action: Any, optimal_move: Any, 
                    new_depth: int, lowest_depth: int) -> Tuple[float, Any, int]:
    """
    Helper function to select the best action for the min node.
    Updates the utility score, the best action, and the lowest depth based on superior values of each.

    Parameters:
        utility_score (float): 
            The utility score of the current action.
        best_score (float): 
            The best utility score found so far.
        action (Any): 
            The current action being evaluated.
        optimal_move (Any): 
            The most optimal action found so far.
        new_depth (int): 
            The depth of the node of the current action.
        lowest_depth (int): 
            The lowest depth of a terminal node found so far.
    
    Returns:
        Tuple[float, Any, int]: 
            A tuple containing the updated best utility score, the optimal move, and the lowest depth of the terminal node.
    """
    if utility_score < best_score: # Max node wants the lowest possible score
        return utility_score, action, new_depth
    elif utility_score == best_score: # Tiebreaker to choose node w/ lower depth
        if new_depth < lowest_depth:
            return utility_score, action, new_depth
        elif new_depth == lowest_depth and action < optimal_move: # Tiebreaker to choose node w/ higher move
            return best_score, action, lowest_depth
    return best_score, optimal_move, lowest_depth
