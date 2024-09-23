
from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
from helpers.utils import NotImplemented

import queue
import heapq
from _heapq import heappush, heappop, heapify


def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    frontier = list()                                                               # step1: create frontier using list
    frontier.append(initial_state)                                                  # step2: add initial state to the frontier
    explored = set()                                                                # step3: create explored set to keep track of explored nodes
    explored.add(initial_state)                                                     # step4: add initial state to the explored set
    parentDict = dict()                                                             # step4:create dictionary to store parent-child pairs for tracing the solution path

    while frontier:                                                                 # step5: loop on frontier

        node = frontier.pop(0)                                                      # step6: get next node in the frontier

        if problem.is_goal(node):                                                   # step7: check if the current node is the goal

            solutionPath = list()                                                   # step8: create a list to store the solution path

            while node is not initial_state:
                node, action = parentDict[node]                                     # step9: trace back the path to goal

                solutionPath.insert(0, action)                                      # step10:to print the solution in Top Down style

            return solutionPath                                                     # step11: return the path

        for action in problem.get_actions(node):                                    # step12: get all possible actions and loop on successors
            successor = problem.get_successor(node, action)

            if successor not in explored:
                frontier.append(successor)                                          # step13: add successor to frontier
                explored.add(successor)                                             # step14: add successor to explored
                parentDict[successor] = (node, action)                              # step15: store parent-child pair for tracing

    return None


def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    frontier = list()                                                           #step1: create frontier using list
    frontier.append(initial_state)                                              #step2: add initial state to the frontier
    explored = set()                                                            #step3: create explored set to keep track of explored nodes
    parentDict = dict()                                                         #step4:create dictionary to store parent-child pairs for tracing the solution path

    while frontier:                                                             #step5: loop on frontier
        node = frontier.pop()                                                   #step6: get next node in the frontier
        if node in explored:
            continue
        explored.add(node)                                                      #step7: add node to explored set

        if problem.is_goal(node):                                               #step8: check if is goal or not

            solution = list()
            while node is not initial_state:
                node, action = parentDict[node]                                 #step9: trace back the path to goal
                solution.insert(0, action)
            return solution

        for action in problem.get_actions(node):                                 #step8: get all possible actions and loop on successors
            successor = problem.get_successor(node, action)
            if successor not in explored:
                frontier.append(successor)                                       #step9: add successor to frontier
                parentDict[successor] = (node, action)                           #step 10: store parent-child pair for tracing

    return None

def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    frontier = queue.PriorityQueue()                                                #step1: create frontier using priority queue
    frontier.put((0, (str(initial_state), initial_state), []))                      #step2: add initial state to the frontier
    explored = set()                                                                #step3: create explored set to keep track of explored nodes

    while not frontier.empty():                                                     #step4: loop on frontier
        cost, node, path = frontier.get()                                           #step5: get next node in the frontier

        if node[1] not in explored:
            if problem.is_goal(node[1]):                                            #step6: check if node is goal or not
                return path

            explored.add(node[1])                                                   #step7: add node to explored set
            actions = problem.get_actions(node[1])                                  #step8: get all possible actions
            for action in actions:                                                  #step9: loop on possible successors
                successor = problem.get_successor(node[1], action)
                frontier.put((cost + problem.get_cost(node[1], action), (str(successor), successor), path + [action]))  #step 10: add the successor to frontier

    return None


def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    priority = 0
    frontier = queue.PriorityQueue()                                                #step1: create frontier using priority queue
    frontier.put((0, (priority, initial_state), []))                                #step2: add initial state to the frontier
    explored = set([])                                                              #step3: create explored set to keep track of explored nodes
    costDict = {initial_state: 0}                                                   #step4: create dictionary to store the cost  {state, cost}

    while not frontier.empty():                                                     #step5: loop on frontier
        h, node, path = frontier.get()                                              #step5: get next node in the frontier

        if node[1] not in explored:
            if problem.is_goal(node[1]):                                            #step6: check if node is goal or not
                return path

            explored.add(node[1])                                                   #step7: add node to explored set

            actions = problem.get_actions(node[1])                                  #step8: get all possible actions
            for action in actions:                                                  #step9: loop on possible successors
                successor = problem.get_successor(node[1], action)
                costDict[successor] = costDict[node[1]] + problem.get_cost(node[1], action)     #step10: calculate cost
                priority += 1
                if successor not in explored:                                       #step11: check successor is not explored and cost is less than
                    frontier.put((costDict[successor] + heuristic(problem, successor), (priority, successor), path + [action]))     #step 12: add the successor to frontier

    return None


def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    # An increasing counter that will be put beside the cost to break the tie
    # in the arrangement of the nodes in the priority queue
    counter = 0

    node = initial_state

    frontier = list()  # Priority Queue for the frontier states

    heappush(frontier,
             (heuristic(problem, node), counter, node))  # add the initial state to the frontier priority queue
    counter += 1

    explored = set()  # Set for the explored states

    parentDict = dict()  # create a dictionary for saving parent and child pairs for each child for tracing the path to the solution

    while frontier:  # loop until all the frontier is explored

        node = heappop(frontier)[2]  # choose the node with the lowest cost

        if problem.is_goal(node):  # check if the current state is the goal

            solutionPath = list()  # create a list to store the solution path

            while node is not initial_state:  # loop on the dictionary to fill the solution path

                node, action = parentDict[node]  # parent_node, action = (node, action)

                solutionPath.insert(0, action)  # to print the solution in Top Down style

            return solutionPath  # return the path

        explored.add(node)  # Add the state to the explored set

        for action in problem.get_actions(node):  # loop on the possible actions of the current state

            child = problem.get_successor(node, action)  # get a child

            is_frontier = False  # check if the child in the frontier
            for front in frontier:
                if child is front[2]:
                    is_frontier = True
                    break

            if child not in explored:  # if the child is not explored

                j = None  # get the child index in the frontier
                for i in range(len(frontier)):
                    if child == (frontier[i])[2]:
                        j = i
                        break

                # if the child in the frontier and the current path is better than the stored path
                if is_frontier and heuristic(problem, child) < frontier[j][0]:
                    # update the child heuristic cost inside the frontier
                    frontier[j] = (heuristic(problem, child), counter, child)
                    parentDict[child] = (node, action)
                    heapify(frontier)  # Reorder the heap
                    counter += 1
                else:
                    # push the child inside the frontier queue
                    parentDict[child] = (node, action)
                    heappush(frontier, (heuristic(problem, child), counter, child))
                    counter += 1

            # add the child to the explored set
            explored.add(child)

    return None