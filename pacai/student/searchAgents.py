"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging
import math

from pacai.core.actions import Actions
# from pacai.core.search import heuristic not used, commented out for style :/
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.distance import manhattan, maze
from pacai.core.directions import Directions
from pacai.student.search import breadthFirstSearch

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        # *** Your Code Here ***
    def isGoal(self, state):

        if state[1] == (1, 1, 1, 1):
            return True
        
        return False

    def startingState(self):
        return (self.startingPosition, (0, 0, 0, 0))
    
    def successorStates(self, state):
        successors = []
        self._numExpanded += 1
        position, visitedCorners = state
        
        for action in Directions.CARDINAL:
            x, y = position
            listCpy = list(visitedCorners)  # list of visited corners
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            # check if the next position hits a wall
            if not self.walls[nextx][nexty]:
                # if the next position is a corner, update the visited corners
                if (nextx, nexty) in self.corners:
                    cornerIndex = self.corners.index((nextx, nexty))
                    listCpy[cornerIndex] = 1  # mark corner as visited
                
                # add new state to successors
                newState = ((nextx, nexty), tuple(listCpy))  # update position and visited corners
                successors.append((newState, action, 1))  # append
                
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem using a slight inflation factor with rounding.
    """
    cornersVisited = state[1]  # visited
    corners = problem.corners  # CORNER COORDINATES!!!
    current_position = state[0]
    dists = []

    # distance to unvisited corners
    for index, corner in enumerate(corners):
        if cornersVisited[index] == 0:  # unvisited corner
            dists.append(manhattan(current_position, corner))

    if not dists:
        return 0  # no cost

    # manipulating max distance, trial and error
    inflated_max_dist = 1.05 * max(dists)

    # rounded value of dist
    return math.floor(inflated_max_dist)

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    # *** Your Code Here ***

    # used a cache to store distances in order to avoid recalcuating, making search more efficient
    pac = state[0]
    foodList = state[1].asList()
    
    if not foodList:
        return 0
    
    #  heuristicInfo for caching distances
    if 'distCache' not in problem.heuristicInfo:
        problem.heuristicInfo['distCache'] = {}
    
    dists = {}
    distCache = problem.heuristicInfo['distCache']
    
    for foodCoordinates in foodList:
        if (pac, foodCoordinates) not in distCache:
            # compute + store manhattan distance
            distCache[(pac, foodCoordinates)] = manhattan(pac, foodCoordinates)
        dists[foodCoordinates] = distCache[(pac, foodCoordinates)]
    
    # max manhattan distance from cache
    maxCoord = max(dists, key=dists.get)
    return maze(pac, maxCoord, problem.startingGameState)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """
        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        return breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        x, y = state
        return self.food[x][y]

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)