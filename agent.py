# Input states for Agent step function
from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation

# State steppers
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step

# Utility helpers
import random
import heapq
import itertools
import math

#helper for state
from dataclasses import dataclass
from typing import Tuple

#Helper class imports
from dataclasses import replace
from dataclasses import dataclass
from typing import Tuple
from collections import deque


#Entity imports
from grid_adventure.entities import (
    AgentEntity,
    FloorEntity,
    WallEntity,
    ExitEntity,
    CoinEntity,
    GemEntity,
    KeyEntity,
    LockedDoorEntity,
    LavaEntity,
    BoxEntity,
    SpeedPowerUpEntity,
    ShieldPowerUpEntity,
    PhasingPowerUpEntity,
)

@dataclass(frozen=True)
class SearchState:
    agent_pos: Tuple[int, int]       # Coordinate of the agent 
    hp: int                          # Agent's current health
    keys_held: int                   # Number of keys in inventory
    keys_left: Tuple[Tuple[int, int], ...]    # Remaining key locations
    gems_left: Tuple[Tuple[int, int], ...]           # Remaining gem locations
    # coins_left: Tuple[Tuple[int, int], ...]          # changed this to be stored in self.coins instead
    doors_locked: Tuple[Tuple[int, int], ...]        # Remaining locked door locations
    boxes_pos: Tuple[Tuple[int, int], ...]           # Current box locations
        
    # Track the duration of powerups currently active -> (("boots", 5), ("shield", 2))
    active_powerups: Tuple[Tuple[str, int], ...]     
        
    # uncollected powerups for fast coordinate checking
    uncollected_boots: Tuple[Tuple[int, int], ...]   # Remaining boots locations
    uncollected_shields: Tuple[Tuple[int, int], ...] # Remaining shield locations
    uncollected_ghosts: Tuple[Tuple[int, int], ...]  # Remaining ghost locations


class Agent:
    """Grid Adventure: Variant 1 agent template.

    This class is the single public interface that Coursemology will import and
    interact with when evaluating your submission. You should extend the
    internals (add helper classes / functions in other files if you wish) but
    MUST preserve:

    1. The class name: Agent
    2. The public method: step(self, state: GridState | ImageObservation) -> Action

    High‑level lifecycle per environment tick:
        state  --->  step(...)  --->  Action

    The "state" object type depends on the task:
    - Task 1: A fully structured GridState instance.
    - Task 2: An ImageObservation dictionary whose primary observation is an RGBA image
      plus limited structured metadata in the 'info' sub‑dict. In this case you
      typically perform perception to build (or approximate) an internal
      structured representation before planning.
    - Task 3: Input state could be either a GridState instance 
      or an ImageObservation dictionary

    Constraints:
    - Keep per‑step latency small (single CPU, ~1GB RAM). Avoid O(W*H) scans of
      the full grid every step.
    - Determinism helps reproducibility; seed your own RNG if you add any
      random components.

    You may add __init__ parameters (with defaults) if needed for your own
    development, but the grader will instantiate Agent() with no arguments.
    """

    def __init__(self):
        """Initialize your agent.

        Put all one‑time setup here (e.g., hardcoded ML model weights, 
        precomputing heuristic tables). Keep it fast and memory‑light 
        to respect platform limits.
        """
        self.actions_to_make = []
        self.has_searched = False

    def step(self, state: GridState | ImageObservation) -> Action:
        #have to search first if we dont have a path
        if not self.has_searched:

            grid = state             
            initial_state = self.extract_initial_state(grid)
            self.a_star(initial_state)
            self.has_searched = True
            
        #pop actions from our action_plan
        if self.actions_to_make:
            next_action = self.actions_to_make.pop(0)
            return next_action
            
        #if no moves available
        return Action.WAIT
    
    def parse(self, observation: ImageObservation) -> GridState:
        """Parse image observation into GridState representation.

        NOTE: This method is optional and intended for debugging in Grid Play only. 
        You do not need to implement it for Coursemology submission, but it can be 
        helpful for visualizing your agent's perception during development. 
        Implementing this method will not affect grading.

        Parameters
        ----------
        observation : ImageObservation
            The raw image observation and metadata from the environment.

        Returns
        -------
        GridState
            The reconstructed internal representation of the environment state.
        """
        # Placeholder: implement perception logic here
        pass
    
    def info(self) -> dict[str, str]:
        """Return info about the agent.

        NOTE: This method is optional and intended for debugging in Grid Play only. 
        You do not need to implement it for Coursemology submission, but it can be 
        helpful for visualizing your agent's internal state during development. 
        Implementing this method will not affect grading.
        """
        # Optional: return info about the agent
        return {"name": "Random AI Agent"}
    
    def precompute_bfs_distances(self, target_pos):
        #BFS to find closest distances with walls in the way for a tighter heuristic
        distances = {target_pos: 0}
        queue = deque([(target_pos, 0)])
        while queue:
            (x, y), dist = queue.popleft()
            for nx, ny in [(x,y-1),(x,y+1),(x-1,y),(x+1,y)]:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.walls and (nx, ny) not in distances:
                        distances[(nx, ny)] = dist + 1
                        queue.append(((nx, ny), dist + 1))
        return distances
    
    def extract_initial_state(self, grid: GridState) -> SearchState:
        # initialise state fields
        self.walls = set()
        self.lava = set()
        self.exit_pos = None
        self.width = grid.width
        self.height = grid.height
        
        # initialise state variables
        agent_pos = None
        gems = []
        coins = []
        doors = []
        boxes = []
        keys = []
        boots = []
        shields = []
        ghosts = []
        starting_hp = None
        
        # scan the grid
        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                entities = grid.objects_at(pos)
                
                for e in entities:
                    #fixed entities
                    if isinstance(e, WallEntity):
                        self.walls.add(pos)
                    elif isinstance(e, LavaEntity):
                        self.lava.add(pos)
                    elif isinstance(e, ExitEntity):
                        self.exit_pos = pos
                        
                    #variable entities
                    elif isinstance(e, AgentEntity):
                        agent_pos = pos
                        starting_hp = e.health.current_health
                    elif isinstance(e, GemEntity):
                        gems.append(pos)
                    elif isinstance(e, CoinEntity):
                        coins.append(pos)
                    elif isinstance(e, LockedDoorEntity):
                        doors.append(pos)
                    elif isinstance(e, BoxEntity):
                        boxes.append(pos)
                    elif isinstance(e, KeyEntity):
                        keys.append(pos)
                    elif isinstance(e, SpeedPowerUpEntity):
                        boots.append(pos)
                    elif isinstance(e, ShieldPowerUpEntity):
                        shields.append(pos)
                    elif isinstance(e, PhasingPowerUpEntity):
                        ghosts.append(pos)

        self.bfs_dist_from = {}
        self.bfs_dist_from[self.exit_pos] = self.precompute_bfs_distances(self.exit_pos)
        for gem_pos in gems:
            self.bfs_dist_from[gem_pos] = self.precompute_bfs_distances(gem_pos)

        self.coin_positions = set(coins) #initially this was stored in every state but it may have caused too big a branching factor
                                         #and led to timeouts
                        
        return SearchState(
            agent_pos=agent_pos,
            hp=starting_hp,
            keys_held=0,
            keys_left=tuple(keys), 
            gems_left=tuple(gems),
            doors_locked=tuple(doors),
            boxes_pos=tuple(sorted(boxes)),
            active_powerups=tuple(),
            uncollected_boots=tuple(boots),
            uncollected_shields=tuple(shields),
            uncollected_ghosts=tuple(ghosts)
        )
    

    def get_successors(self, state: SearchState):
        x, y = state.agent_pos

        ticked_powerups = tuple(
            (name, dur) if name == "ShieldPowerUpEntity" else (name, dur - 1)
            for name, dur in state.active_powerups
            if name == "ShieldPowerUpEntity" or dur > 1
        )

        has_speed = any(n == "SpeedPowerUpEntity" for n, _ in state.active_powerups)
        has_ghost = any(n == "PhasingPowerUpEntity" for n, _ in state.active_powerups)
        doors_set = set(state.doors_locked)
        boxes_set = set(state.boxes_pos)

        successors = []
        successors.extend(self.pickup_successors(state, x, y, ticked_powerups))
        successors.extend(self.movement_successors(state, x, y, ticked_powerups, has_speed, has_ghost, doors_set, boxes_set))
        successors.extend(self.use_key_successors(state, x, y, ticked_powerups, doors_set))
        return successors
    
    def pickup_successors(self, state, x, y, ticked):
        if (x, y) in state.keys_left:
            return [(replace(state,
                keys_held=state.keys_held + 1,
                keys_left=tuple(k for k in state.keys_left if k != (x, y)),
                active_powerups=ticked), Action.PICK_UP)]

        if (x, y) in state.gems_left:
            return [(replace(state,
                gems_left=tuple(g for g in state.gems_left if g != (x, y)),
                active_powerups=ticked), Action.PICK_UP)]

        if (x, y) in state.uncollected_boots:
            return [(replace(state,
                uncollected_boots=tuple(b for b in state.uncollected_boots if b != (x, y)),
                active_powerups=tuple(sorted(ticked + (("SpeedPowerUpEntity", 5),)))),
                Action.PICK_UP)]

        if (x, y) in state.uncollected_shields:
            return [(replace(state,
                uncollected_shields=tuple(s for s in state.uncollected_shields if s != (x, y)),
                active_powerups=tuple(sorted(ticked + (("ShieldPowerUpEntity", 5),)))),
                Action.PICK_UP)]

        if (x, y) in state.uncollected_ghosts:
            return [(replace(state,
                uncollected_ghosts=tuple(g for g in state.uncollected_ghosts if g != (x, y)),
                active_powerups=tuple(sorted(ticked + (("PhasingPowerUpEntity", 5),)))),
                Action.PICK_UP)]
        
        #initially, there was another if statement for coins but i decided to make my agent ignore coins and only pick the coins up 
        #if it was along the way of the optimal path to the exit. 

        return []
    
    def movement_successors(self, state, x, y, ticked, has_speed, has_ghost, doors_set, boxes_set):
        results = []
        directions = {
            Action.UP: (0, -1), Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0), Action.RIGHT: (1, 0),
        }

        for action, (dx, dy) in directions.items():
            max_steps = 1
            if has_speed :
                max_steps = 2
            final_pos = (x, y)
            cur_boxes = list(state.boxes_pos)
            cur_box_set = set(cur_boxes)
            path = []  # all tiles stepped on

            for step in range(1, max_steps + 1):
                nx, ny = x + dx * step, y + dy * step

                # boundary
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    break

                if has_ghost:
                    final_pos = (nx, ny)
                    path.append((nx, ny))
                    continue

                # wall or locked door
                if (nx, ny) in self.walls or (nx, ny) in doors_set:
                    break

                # box pushing
                if (nx, ny) in cur_box_set:
                    bx, by = nx + dx, ny + dy
                    if not (0 <= bx < self.width and 0 <= by < self.height): #if we break out of the boundary
                        break
                    if (bx, by) in self.walls or (bx, by) in doors_set or (bx, by) in cur_box_set: #if we are pushing into another box or locked door
                        break
                    cur_boxes = [(bx, by) if b == (nx, ny) else b for b in cur_boxes]
                    cur_box_set = cur_box_set - {(nx, ny)} | {(bx, by)}

                final_pos = (nx, ny)
                path.append((nx, ny))

            # did not move
            if final_pos == (x, y):
                continue

            # apply lava damage if applicable
            next_hp = state.hp
            final_powerups = list(ticked)
            if not has_ghost:
                for pos in path:
                    if pos in self.lava:
                        shield_found = False
                        for i, (name, uses) in enumerate(final_powerups):
                            if name == "ShieldPowerUpEntity":
                                shield_found = True
                                if uses > 1:
                                    final_powerups[i] = (name, uses - 1)
                                else:
                                    final_powerups.pop(i)
                                break
                        if not shield_found:
                            next_hp -= 2
                        if next_hp <= 0:
                            break

            if next_hp <= 0:
                continue

            next_state = replace(state,
                agent_pos=final_pos, hp=next_hp,
                boxes_pos=tuple(sorted(cur_boxes)),
                active_powerups=tuple(sorted(final_powerups)))
            results.append((next_state, action))

        return results
    
    def use_key_successors(self, state, x, y, ticked, doors_set):
        if state.keys_held <= 0:
            return []

        to_unlock = []
        keys_available = state.keys_held
        for pos in [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if pos in doors_set and keys_available > 0:
                to_unlock.append(pos)
                keys_available -= 1

        if not to_unlock:
            return []

        unlock_set = set(to_unlock)
        return [(replace(state,
            keys_held=state.keys_held - len(to_unlock),
            doors_locked=tuple(d for d in state.doors_locked if d not in unlock_set),
            active_powerups=ticked), Action.USE_KEY)]
    

    def heuristic(self, state: SearchState) -> int:
        #if gems left is empty and we are at the goal, heuristic should return 0
        if not state.gems_left and state.agent_pos == self.exit_pos:
            return 0
        
        def get_dist(p1, p2): #returns distance between p1 and p2
            if p1 == p2:
                return 0
            if p2 in self.bfs_dist_from and p1 in self.bfs_dist_from[p2]:
                return self.bfs_dist_from[p2][p1]
            if p1 in self.bfs_dist_from and p2 in self.bfs_dist_from[p1]:
                return self.bfs_dist_from[p1][p2]
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        if state.gems_left: #if we have gems left, return the (distance of agent and gem closest to agent) + (distance of exit and gem closest to exit)
            dist_agent_to_gem = min(get_dist(state.agent_pos, g) for g in state.gems_left)
            dist_gem_to_exit = min(get_dist(g, self.exit_pos) for g in state.gems_left)
            total_dist = dist_agent_to_gem + dist_gem_to_exit
        else:
            total_dist = get_dist(state.agent_pos, self.exit_pos) #if no gems left then return dist to exit

        return math.ceil(total_dist / 2) * 3 #assume we got boots active
    
    def a_star(self, initial_state: SearchState):
        """
        Executes the A* search algorithm to find the optimal path to the exit.
        """
        #use itertools as a tiebreaker if two states have same score
        counter = itertools.count() 
        
        # frontier stores tuples of (f(n), tie_breaker, state)
        frontier = []
        heapq.heappush(frontier, (self.heuristic(initial_state), next(counter), initial_state))
        
        # g(n) = cost
        gn = {initial_state: 0}
        
        # keep track of our moves
        came_from = {}

        #visited set
        visited = {}
        
        while frontier:
            # pop the state with the lowest f(n)
            f, _, current_state = heapq.heappop(frontier)

            #check if visited
            if current_state in visited and visited[current_state] <= gn[current_state]:
                continue
            visited[current_state] = gn[current_state]
            
            # if collected all gems and found exit,
            if len(current_state.gems_left) == 0 and current_state.agent_pos == self.exit_pos:
                # form our moves in order
                self.actions_to_make = self.reconstruct_path(came_from, current_state)
                return 
                
            #expand the tree
            for next_state, action in self.get_successors(current_state):

                #calculate the true cost of this action
                action_cost = self.calculate_action_cost(current_state, next_state, action)
                new_gn = gn[current_state] + action_cost

                #update and push to the queue if this is a strictly cheaper path
                if new_gn < gn.get(next_state, float('inf')) :
                    came_from[next_state] = current_state, action
                    gn[next_state] = new_gn
                    
                    # calculate f(n) and add to frontier
                    fn = new_gn + self.heuristic(next_state)
                    heapq.heappush(frontier, (fn, next(counter), next_state))
                    
        # if we never find the exit
        self.actions_to_make = []


    def reconstruct_path(self, came_from, current_state):
        pairs = []
        while current_state in came_from:
            prev_state, action = came_from[current_state]
            pairs.append((prev_state, action, current_state))
            current_state = prev_state
        pairs.reverse()

        final_path = []
        remaining_coins = set(self.coin_positions)
        for prev_state, action, next_state in pairs:
            final_path.append(action)
            if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                landed = next_state.agent_pos
                if landed in remaining_coins:
                    final_path.append(Action.PICK_UP)
                    remaining_coins.discard(landed)

        return final_path
    
    def calculate_action_cost(self, current_state: SearchState, next_state: SearchState, action: Action) -> int:
        cost = 3  # every turn incurs a base cost of 3
            
        return cost