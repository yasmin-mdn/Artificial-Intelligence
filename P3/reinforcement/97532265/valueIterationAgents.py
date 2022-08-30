# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from sre_parse import State
from xmlrpc.client import MININT
import mdp, util
import sys

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
    

    def runValueIteration(self):
        for iter in range(0,self.iterations,1):
            update_values = self.values.copy()
            for state in self.mdp.getStates():
                Qvalues = [float('-inf')]
                if self.mdp.isTerminal(state):
                    update_values[state] = 0
                else:
                    legal_actions = self.mdp.getPossibleActions(state)
                    for action in legal_actions:
                        Qvalues.append(self.getQValue(state, action))
                    update_values[state] = max(Qvalues)

            self.values = update_values
 

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Transitions= self.mdp.getTransitionStatesAndProbs(state, action)
        value_utilities = []
        for transition in Transitions:
            Prob = transition[1]
            statePrime = transition[0]
            dis_fact = self.discount
            reward = self.mdp.getReward(state, action, statePrime)
            ut=Prob * (reward + (dis_fact * self.values[statePrime]))
            value_utilities.append(ut) 
        return sum(value_utilities)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        
        actions = self.mdp.getPossibleActions(state)
        q_value = MININT
        next_action = None
        for action in actions:
            value = self.computeQValueFromValues(state, action)
            if value > q_value:
                q_value = value
                next_action = action

        return next_action
       

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        States = self.mdp.getStates()
        i_Iterator = 0
        for iter in range(0, self.iterations):
            if i_Iterator == len(States): i_Iterator = 0
            targetState = States[i_Iterator]
            i_Iterator += 1
            if self.mdp.isTerminal(targetState):
                continue
            bestAction = self.computeActionFromValues(targetState)
            QValue = self.computeQValueFromValues(targetState,bestAction)
            self.values[targetState] = QValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                maxVal = MININT
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState]={state}
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                maxQ = MININT
                for action in self.mdp.getPossibleActions(s):
                    Q = self.computeQValueFromValues(s ,action)
                    if Q>maxQ:
                        maxQ = Q
                diff = abs(maxQ - self.values[s])
                pq.update(s, -diff)
                
        for _ in range(self.iterations):

            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                maxVal = MININT
                for action in self.mdp.getPossibleActions(s):
                    Q = self.computeQValueFromValues(s ,action)
                    if Q>maxVal:
                        maxVal = Q
                self.values[s] = maxVal

            for p in predecessors[s]:
                maxQ = MININT
                for action in self.mdp.getPossibleActions(p):
                    Q = self.computeQValueFromValues(p ,action)
                    if Q>maxQ:
                        maxQ = Q
                diff = abs(maxQ - self.values[p])
                if diff>self.theta:
                    pq.update(p, -diff)

