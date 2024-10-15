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
    ```
    print("Start: %s" % (str(problem.startingStaet())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """



    # *** Your Code Here ***
    fringe = Stack()
    seen = set()
    

    fringe.push((problem.startingState(), [], 0))
    while not fringe.isEmpty():
        curr, moves, cost = fringe.pop()  
        if (problem.isGoal(curr)):
            return moves
        if curr not in seen:
            seen.add(curr)
            succ = problem.successorStates(curr)
            for (next, move, noCost) in succ:  
                if next not in seen:
                    nextMoves = moves + [move]
                    fringe.push((next, nextMoves, noCost))

    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    fringe = Queue()
    seen = set()
    fringe.push((problem.startingState(), [], 0))
    while not fringe.isEmpty():
        curr, moves, cost = fringe.pop() 
        if (problem.isGoal(curr)):
            return moves
        if curr not in seen:
            seen.add(curr)
            succ = problem.successorStates(curr)
            for (next, move, noCost) in succ:  
                nextMoves = moves + [move]
                fringe.push((next, nextMoves, noCost))
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    seen = set()
    fringe = PriorityQueue()

    fringe.push((problem.startingState(), []), 0)
    while not fringe.isEmpty():
        curr, moves = fringe.pop()  
        if (problem.isGoal(curr)):
            return moves
        if curr not in seen:
            seen.add(curr)
            succ = problem.successorStates(curr)
            for (next, move, cost) in succ:
                if next not in seen:
                    nextMoves = moves + [move]
                    fringe.push((next, nextMoves), cost)
    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    fringe = PriorityQueue()
    seen = set()
    # *** Your Code Here ***
    fringe.push((problem.startingState(), [], 0), 0)
    while not fringe.isEmpty():
        curr, moves, currCost = fringe.pop()  # store information at the front of the fringe
        if (problem.isGoal(curr)):
            return moves
        if curr not in seen:
            seen.add(curr)
            succ = problem.successorStates(curr)
            for (next, move, cost) in succ:
                if next not in seen:
                    nextMoves = moves + [move]
                    # iterate on previous calculation to contain current cost
                    costSoFar = currCost + cost + heuristic(next, problem)

                    fringe.push((next, nextMoves, costSoFar), costSoFar)

    return []
