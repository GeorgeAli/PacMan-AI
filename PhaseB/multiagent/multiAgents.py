# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        food = newFood.asList()
        score = successorGameState.getScore()
        ghost_distance = manhattanDistance(currentGameState.getGhostPosition(1), newPos)

        # award packman for staying away from the ghost
        if ghost_distance > 2:
            score += 15

        # award for taking food
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 15

        # technically punish him for not going towards food
        # this just stops pacman from wandering around aimlessly
        score -= 2 * manhattanDistance(min(food, default=[0, 0]), newPos)
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.
      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def Minimax(gameState, agent, depth):

            tree = []
            temp_agent = agent + 1
            total_agents = gameState.getNumAgents() - 1
            legal_actions = gameState.getLegalActions(agent)

            if len(legal_actions) == 0 or depth == self.depth:
                return [self.evaluationFunction(gameState), -1]

            if agent == total_agents:
                temp_agent = 0
                depth += 1

            for every_action in legal_actions:

                move = Minimax(
                    gameState.generateSuccessor(agent, every_action), temp_agent, depth
                )

                if not tree:
                    tree.append(move[0])
                    tree.append(every_action)
                else:

                    previous_min_move = tree[0]

                    if agent == 0 and move[0] > previous_min_move:
                        tree[agent], tree[agent + 1] = move[agent], every_action
                    elif agent != 0 and move[0] < previous_min_move:
                        tree[0], tree[1] = move[0], every_action
            return tree

        resulting_tree = Minimax(gameState, 0, 0)

        return resulting_tree[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth, a, b):

            legal_actions = gameState.getLegalActions(0)
            value_to_compare = -float("inf")
            best_action = -1

            if len(legal_actions) == 0 or depth == self.depth:
                return (self.evaluationFunction(gameState), -1)

            for every_action in legal_actions:

                move = min_value(
                    gameState.generateSuccessor(0, every_action), 1, depth, a, b
                )

                if value_to_compare < move[0]:
                    value_to_compare, best_action = move[0], every_action

                if value_to_compare > b:
                    return value_to_compare, -1

                a = max(a, value_to_compare)

            return value_to_compare, best_action

        def min_value(gameState, agent, depth, a, b):

            legal_actions = gameState.getLegalActions(agent)
            value_to_compare = float("inf")
            best_action = -1

            if len(legal_actions) == 0:
                return self.evaluationFunction(gameState), -1

            for every_action in legal_actions:

                if agent == gameState.getNumAgents() - 1:
                    move = max_value(
                        gameState.generateSuccessor(agent, every_action),
                        depth + 1,
                        a,
                        b,
                    )
                else:
                    move = min_value(
                        gameState.generateSuccessor(agent, every_action),
                        agent + 1,
                        depth,
                        a,
                        b,
                    )

                if move[0] < value_to_compare:
                    value_to_compare, best_action = move[0], every_action

                if value_to_compare < a:
                    return value_to_compare, best_action

                b = min(b, value_to_compare)

            return value_to_compare, best_action

        return max_value(gameState, 0, -float("inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):

            legal_actions = gameState.getLegalActions(0)
            value_to_compare = -float("inf")
            best_action = -1

            if len(legal_actions) == 0 or depth == self.depth:
                return self.evaluationFunction(gameState), -1

            for every_action in legal_actions:
                move = probable_action(
                    gameState.generateSuccessor(0, every_action), 1, depth
                )
                if value_to_compare < move[0]:
                    value_to_compare, best_action = move[0], every_action
            return value_to_compare, best_action

        def probable_action(gameState, agentID, depth):

            legal_actions = gameState.getLegalActions(agentID)
            lenth_of_la = float(len(legal_actions))
            value_to_compare = 0

            if lenth_of_la == 0:
                return (self.evaluationFunction(gameState), -1)

            for every_action in legal_actions:
                if agentID == gameState.getNumAgents() - 1:
                    sucsValue = max_value(
                        gameState.generateSuccessor(agentID, every_action), depth + 1
                    )
                else:
                    sucsValue = probable_action(
                        gameState.generateSuccessor(agentID, every_action),
                        agentID + 1,
                        depth,
                    )
                prob = sucsValue[0] / lenth_of_la
                value_to_compare += prob

            return value_to_compare, -1

        to_return = max_value(gameState, 0)[1]

        print(to_return)

        return to_return


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: Look for the comments above the lines
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"

    food = currentGameState.getFood().asList()
    food_length = len(food)
    ghost = currentGameState.getGhostStates()
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    # Decrease the score when a capsule is taken
    # Forces pacman to want as less capsules as possible
    score -= 300 * len(currentGameState.getCapsules())

    score -= 30 * food_length

    # Calculate distances of food
    food_distance = min(
        (manhattanDistance(pos, every_food) for every_food in food), default=0
    )

    # Decrease points for not going closer to food
    # Forces pacman to move towards food and not thrash around
    score -= 1 * food_distance

    found_ghost = 0
    ghost_distances = []
    for every_ghost in range(len(ghost)):
        if ghost[every_ghost].scaredTimer:
            found_ghost = 1
            ghost_distances.append(manhattanDistance(
                pos, ghost[every_ghost].getPosition()
            ))

    if found_ghost:
        ghost_min = min(range(len(ghost_distances)), key=ghost_distances.__getitem__)
        ghost_min_distance = manhattanDistance(pos, ghost[ghost_min].getPosition())
        score -= 2 * ghost_min_distance

    return score


# Abbreviation
better = betterEvaluationFunction
