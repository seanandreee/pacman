"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.priorityQueue import PriorityQueue
from pacai.util.queue import Queue

def depthFirstSearch(problem):

    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
print("Start: %s" % (str(problem.startingStaet())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))

    """

    # *** Your Code Here ***
    visited = set()
    fringe = Stack()

    fringe.push((problem.startingState(), [], 0))

    while not fringe.isEmpty():
        current_state, actions, cost = fringe.pop()

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoal(current_state):
            return actions

        for state, action, cost in problem.successorStates(current_state):
            if state not in visited:
                new_actions = actions + [action]
                fringe.push((state, new_actions, 0))  # cost is irrelevant in DFS

    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    fringe = Queue()  # frontier for bfs is queue
    visited = set()
    fringe.push((problem.startingState(), [], 0))

    while not fringe.isEmpty():  # iterate until goal is reached or all possibilities exhausted
        curr, moves, cost = fringe.pop()

        if problem.isGoal(curr):  # return moves leading to the goal
            return moves

        if curr not in visited:  # expand unexplored nodes
            visited.add(curr)

            for next_state, action, cost in problem.successorStates(curr):
                if next_state not in visited:
                    updated_moves = moves + [action]
                    fringe.push((next_state, updated_moves, 0))

    return []  # return empty list if no solution is found


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    fringe = PriorityQueue()  # frontier is priority q
    cost_so_far = {}  # dictionary for lowest cost

    start_state = problem.startingState()
    fringe.push((start_state, [], 0), 0)
    cost_so_far[start_state] = 0

    while not fringe.isEmpty():
        curr, moves, currCost = fringe.pop()

        if problem.isGoal(curr):  # return moves for lowest goal
            return moves

        for next_state, move, step_cost in problem.successorStates(curr):  # expand nodes
            new_cost = currCost + step_cost

            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                next_moves = moves + [move]
                fringe.push((next_state, next_moves, new_cost), new_cost)

    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first (A* search).
    """
    fringe = PriorityQueue()
    visited = {}
    start_state = problem.startingState()
    # push the start state with cost 0 and heuristic
    fringe.push((start_state, [], 0), heuristic(start_state, problem))
    visited[start_state] = 0

    while not fringe.isEmpty():
        curr, moves, currCost = fringe.pop()
        # check if goal state reached
        if problem.isGoal(curr):
            return moves
        for next_state, move, step_cost in problem.successorStates(curr):
            new_cost = currCost + step_cost
            # only consider this successor if it's not visited or has a lower cost path
            if next_state not in visited or new_cost < visited[next_state]:
                visited[next_state] = new_cost  # update cost
                priority = new_cost + heuristic(next_state, problem)
                fringe.push((next_state, moves + [move], new_cost), priority)

    return []
