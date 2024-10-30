import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.recentPositions = []

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        currentPos = gameState.getPacmanPosition()
        self.recentPositions.append(currentPos)
        if len(self.recentPositions) > 5:  # Keep the last 5 positions only
            self.recentPositions.pop(0)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Evaluation function that scores moves with high rewards for food collection and
        minimal penalties for ghost proximity, while discouraging repetitive moves.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        oldPosition = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        oldFood = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        score = successorGameState.getScore()  # Base score

        # Reward for eating food pellets
        for food in oldFood:
            if food not in newFood:  # Food was eaten
                score += 10

        # Reward for moving closer to food
        if newFood:  # Ensure there's still food to evaluate
            oldFoodMinDist = min([manhattan(oldPosition, food) for food in oldFood])
            newFoodMinDist = min([manhattan(newPosition, food) for food in newFood])
            if newFoodMinDist < oldFoodMinDist:  # Moving closer to food
                score += 5

        # Penalize proximity to active ghosts
        for ghostIdx in range(len(newGhostStates)):
            ghost = newGhostStates[ghostIdx]
            ghostDistance = manhattan(newPosition, ghost.getPosition())

            if ghost.isBraveGhost() and ghostDistance < 6:  # Only penalize active ghosts
                score -= 20

        # Reward for moving closer to scared ghosts
        for ghostIdx in range(len(newGhostStates)):
            ghost = newGhostStates[ghostIdx]
            ghostDistance = manhattan(newPosition, ghost.getPosition())
            scareTime = newScaredTimes[ghostIdx]

            if not ghost.isBraveGhost():  # Ghost is scared
                if scareTime < 5:  # Minor penalty if the ghost will soon recover
                    score -= 5
                elif ghostDistance < manhattan(oldPosition, ghost.getPosition()):
                    score += 1

        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def minimax(self, state, depth, agentIdx):
        # Terminal state
        if state.isWin() or state.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state), None

        # Maximizing function
        if agentIdx == 0:
            maxValue = -float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIdx):
                if action == Directions.STOP:  # Ignoring Directions.STOP as instructed
                    continue
                successor = state.generateSuccessor(agentIdx, action)
                value, _ = self.minimax(successor, depth, agentIdx + 1)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
            return maxValue, bestAction
        else:
            # Minimizing function
            minValue = float('inf')
            bestAction = None
            nextAgentIdx = agentIdx + 1
            if nextAgentIdx >= state.getNumAgents():
                nextAgentIdx = 0  # Pacman goes after the last ghost
                depth += 1       # Increase depth when all agents have moved

            for action in state.getLegalActions(agentIdx):
                if action == Directions.STOP:
                    continue
                successor = state.generateSuccessor(agentIdx, action)
                value, _ = self.minimax(successor, depth, nextAgentIdx)
                if value < minValue:
                    minValue = value
                    bestAction = action
            return minValue, bestAction

    def getAction(self, state):
        _, action = self.minimax(state, 0, 0)
        return action

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getAction(self, state):
        alpha = -float('inf')
        beta = float('inf')
        # Run alpha-beta and only get the action part (ignore the score).
        _, bestMove = self.alphabeta(state, depth=0, agentIdx=0, alpha=alpha, beta=beta)
        return bestMove  # Only return the action, not the tuple

    def alphabeta(self, state, depth, agentIdx, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state), None

        numAgents = state.getNumAgents()
        nextAgentIdx = (agentIdx + 1) % numAgents
        nextDepth = depth + 1 if nextAgentIdx == 0 else depth

        # Pac-Man's turn (maximizing)
        if agentIdx == 0:
            bestScore = -float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIdx):
                successor = state.generateSuccessor(agentIdx, action)
                score, _ = self.alphabeta(successor, nextDepth, nextAgentIdx, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                # Update alpha and prune if possible
                alpha = max(alpha, bestScore)
                if alpha >= beta:
                    break  # Beta cut-off
            return bestScore, bestAction

        # Ghost's turn (minimizing)
        else:
            bestScore = float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIdx):
                successor = state.generateSuccessor(agentIdx, action)
                score, _ = self.alphabeta(successor, nextDepth, nextAgentIdx, alpha, beta)
                if score < bestScore:
                    bestScore = score
                    bestAction = action
                # Update beta and prune if possible
                beta = min(beta, bestScore)
                if alpha >= beta:
                    break  # Alpha cut-off
            return bestScore, bestAction
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        _, bestMove = self.expectimax(state, depth=0, agentIdx=0)
        return bestMove

    def expectimax(self, state, depth, agentIdx):
        if state.isWin() or state.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state), None

        numAgents = state.getNumAgents()
        nextAgentIdx = (agentIdx + 1) % numAgents
        nextDepth = depth + 1 if nextAgentIdx == 0 else depth

        # Pac-Man’s Turn (Maximizing)
        if agentIdx == 0:
            bestScore = -float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIdx):
                successor = state.generateSuccessor(agentIdx, action)
                score, _ = self.expectimax(successor, nextDepth, nextAgentIdx)
                if score > bestScore:
                    bestScore, bestAction = score, action
            return bestScore, bestAction

        # Ghost’s Turn (Chance Node)
        else:
            totalScore = 0
            actions = state.getLegalActions(agentIdx)
            probability = 1 / len(actions)  # Uniform probability for random ghost actions.
            for action in actions:
                successor = state.generateSuccessor(agentIdx, action)
                score, _ = self.expectimax(successor, nextDepth, nextAgentIdx)
                totalScore += probability * score  # Weighted average
            return totalScore, None  # Chance nodes don’t need to return an action

def betterEvaluationFunction(currentGameState):
    """
    An extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    Description:
    This was actually the first implementation I had of ReflexAgent.evalueationFunction(),
    which didn't score full points on the autograder so I decided not to use it so here it is.
    It utilizes a basic exponential system which heavily rewards PacMan for moving towards
    food and eating it, moving towards scared ghosts and eating them, moving towards
    power pellets and eating them, but punishes PacmMan for being near ghosts.
    """

    # Extract useful information from the current game state.
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    
    # Initialize score with the current game score
    score = currentGameState.getScore()

    # 1. Strong Reward for Moving Toward Food
    if newFood:
        minFoodDist = min([manhattan(newPos, food) for food in newFood])
        score += 50.0 / (minFoodDist + 1)  # Strong reward for getting closer to food.

    # 2. Reduced Penalty for Ghost Avoidance
    for ghostState in newGhostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattan(newPos, ghostPos)
        
        if ghostState.isBraveGhost() and ghostDist < 3:  # Only avoid ghosts when very close.
            score -= 2 / (ghostDist + 1)  # Minimal penalty for ghost proximity.

        elif not ghostState.isBraveGhost() and ghostDist < 3:  # Reward chasing scared ghosts.
            score += 100 / (ghostDist + 1)  # Very strong reward for reaching scared ghosts.

    # 3. Stronger Reward for Power Pellet Collection
    capsules = currentGameState.getCapsules()
    if capsules:
        minCapsuleDist = min([manhattan(newPos, capsule) for capsule in capsules])
        score += 60.0 / (minCapsuleDist + 1)  # Reward for moving closer to power pellets.

    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
