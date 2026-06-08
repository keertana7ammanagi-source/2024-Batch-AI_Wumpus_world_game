from dataclasses import dataclass, field
from typing import Optional
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              WUMPUS WORLD — Complete Python / Pygame Implementation          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE OVERVIEW                                                       ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║  World          — Grid data, entity placement, percept computation.          ║
║                   Pure data; no UI, no agent state.                          ║
║                                                                              ║
║  Agent          — Tracks position, inventory, score, alive/won status.       ║
║                   apply_move() validates adjacency and fires events.         ║
║                                                                              ║
║  KnowledgeBase  — Stores what the agent *knows*: visited cells, safe cells,  ║
║                   risky cells, percept history, inferred Wumpus location.    ║
║                                                                              ║
║  InferenceEngine— Six propositional-logic rules that update the KB after     ║
║                   each observation.                                          ║
║                                                                              ║
║  AIPlayer       — BFS-based navigator that uses the KB to pick safe moves.   ║
║                   Priority: grab gold → go home → explore safe cells.        ║
║                                                                              ║
║  Game           — Controller: wires World + Agent + KB + AI together,        ║
║                   handles move() calls, maintains the event/action log.      ║
║                                                                              ║
║  UI (Renderer)  — All Pygame drawing: grid, cells, panels, stats bar, etc.   ║
║                                                                              ║
║  main()         — Pygame event loop: handles input, drives AI timer,         ║
║                   calls Renderer.draw() every frame.                         ║
║                                                                              ║
║  HOW CLASSES INTERACT                                                        ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║  main() ──► Game.move(target)                                                ║
║                ├─► Agent.apply_move()  [validates, updates pos/score]        ║
║                ├─► World.get_percepts()  [computes stench/breeze/glitter]    ║
║                ├─► KnowledgeBase.record()  [saves percept at cell]           ║
║                └─► InferenceEngine.infer(kb)  [fires all 6 rules]            ║
║                                                                              ║
║  main() ──► Game.ai_step()                                                   ║
║                └─► AIPlayer.next_move(agent, world, kb)  [BFS planning]      ║
║                        └─► Game.move(target)  [same path as human]           ║
║                                                                              ║
║  main() ──► Renderer.draw(game)  [reads game state, draws everything]        ║
╚══════════════════════════════════════════════════════════════════════════════╝

CONTROLS:
    Mouse click     Move agent to that cell (must be adjacent)
    Arrow keys      Move agent (Up/Down/Left/Right)
    W A S D         Move agent (alternative)
    R               New game
    V               Toggle "Show All" (reveal Wumpus, pits, gold)
    K               Toggle inference overlay (safe ✓ / risky markers)
    A               Toggle AI auto-play
    Space           Single AI step
    Q / Escape      Quit

  ── Arrow Shooting (NEW) ────────────────────────────────────────────────────
    I               Shoot arrow UP
    K (shoot)       Shoot arrow DOWN    [hold Shift to distinguish from K toggle]
    J               Shoot arrow LEFT
    L               Shoot arrow RIGHT

    Note: Shooting keys use I/J/L + Shift+Arrow to avoid conflicting with
    existing controls. Plain K still toggles the inference overlay.
    Use Shift+Up / Shift+Down / Shift+Left / Shift+Right to shoot.

  ── Bump Percept (UPDATED) ──────────────────────────────────────────────────
    Bump now correctly fires only when the agent attempts to walk OUT OF
    BOUNDS (off the grid edge), matching classic Wumpus World semantics.
    Mouse clicks on non-adjacent cells are silently ignored (no bump).

CHANGELOG (v2):
    + Feature 1: Full arrow shooting system (Agent.shoot_arrow, Game.shoot)
    + Feature 2: Bump percept fires on out-of-bounds moves only
    + Arrow status in stats bar (ARROW: ✓ / ✗)
    + Arrow flash animation on shoot
    + Dead Wumpus shown as ✝ in reveal mode
    + Stench clears automatically after Wumpus death
    + KB wumpus_loc cleared on confirmed kill; risky set updated
"""

import sys
import time
import random
import pygame
from collections import deque


GRID_SIZE = 4          # 4×4 world 
FPS       = 60
AI_SPEED  = 0.70       # seconds between AI auto-steps


# ─── Scoring  ───
SCORE_STEP   = -1
SCORE_GOLD   = +1000
SCORE_CLIMB  = +500
SCORE_DEATH  = -1000
SCORE_ARROW  = -10


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: World
# ═══════════════════════════════════════════════════════════════════════════════
class World:
    """
    Represents the static cave environment.

    Responsibilities:
    - Randomly place the Wumpus, Gold, and Pits at game start.
    - Compute percepts (stench, breeze, glitter) for any cell.
    - Report whether a cell is immediately deadly.

    This class holds ONLY world data — no agent state, no UI.

    Placement rules :
    - Cell (0,0) is always safe (agent start).
    - Wumpus and Gold placed at two distinct random non-start cells.
    - Pits added to remaining cells at 20% probability, minimum 2, maximum 4.
    - Wumpus alive flag allows for future arrow-kill mechanics.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        # All cells except the start
        candidates = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if not (r == 0 and c == 0)
        ]
        random.shuffle(candidates)

        self.wumpus: tuple[int, int]      = candidates[0]
        self.gold:   tuple[int, int]      = candidates[1]
        self.pits:   set[tuple[int, int]] = set()
        self.wumpus_alive: bool           = True

        # Add pits to remaining candidates (20% chance each, min 2, max 4)
        pit_candidates = candidates[2:]
        for cell in pit_candidates:
            if len(self.pits) >= 4:
                break
            if random.random() < 0.20:
                self.pits.add(cell)
        # Guarantee at least 2 pits
        for cell in pit_candidates:
            if len(self.pits) >= 2:
                break
            self.pits.add(cell)

    # ── Grid helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def neighbors(r: int, c: int) -> list[tuple[int, int]]:
        """Return valid orthogonal neighbors of (r, c)."""
        return [
            (nr, nc)
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
        ]

    # ── Percept computation ───────────────────────────────────────────────────

    def get_percepts(self, r: int, c: int, agent_has_gold: bool) -> dict:
        """
        Compute all three sensory percepts at cell (r, c).

        stench  — True if Wumpus is alive and in (r,c) or any neighbor.
        breeze  — True if any pit is in (r,c) or any neighbor.
        glitter — True if gold is here and the agent hasn't picked it up yet.
        """

        wr, wc = self.wumpus
        stench = self.wumpus_alive and (
            (r == wr and c == wc) or
            any(nr == wr and nc == wc for nr, nc in self.neighbors(r, c))
        )
        breeze = (
            (r, c) in self.pits or
            any((nr, nc) in self.pits for nr, nc in self.neighbors(r, c))
        )
        glitter = (not agent_has_gold) and (r, c) == self.gold

        return {"stench": stench, "breeze": breeze, "glitter": glitter}

    def is_deadly(self, r: int, c: int) -> Optional[str]:
        """
        Check if stepping into (r, c) kills the agent.
        Returns 'wumpus', 'pit', or None.
        """
        if self.wumpus_alive and (r, c) == self.wumpus:
            return "wumpus"
        if (r, c) in self.pits:
            return "pit"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: Agent
# ═══════════════════════════════════════════════════════════════════════════════
class Agent:
    """
    Represents the agent's mutable state and actions.

    Responsibilities:
    - Track current position, alive status, gold possession, score.
    - Validate and apply moves (wall-boundary bump check).
    - Shoot the single arrow in a cardinal direction (NEW).
    - Track bump and scream flags for percept reporting.
    - Calculate scoring according to AIMA rules.

    apply_move() is the single entry point for all movement.
    It returns an event dict that Game uses to update logs and status.

    Event types returned by apply_move():
        "move"   — successful move, no hazard
        "death"  — stepped into Wumpus or pit
        "gold"   — picked up gold (first time entering gold cell)
        "win"    — returned to (0,0) with gold
        "bump"   — agent tried to move outside the grid boundary (wall hit)
        "noop"   — game already over, ignored

    Event types returned by shoot_arrow():
        "kill"     — arrow hit and killed the Wumpus
        "miss"     — arrow missed (Wumpus not in line of fire)
        "no_arrow" — no arrow remaining

    Scoring :
        -1  per step
        -10 to shoot the arrow (regardless of hit/miss)
        +1000 for picking up gold
        +500  for climbing out with gold
        -1000 for dying

    BUMP SEMANTICS (v2 — matches classic AIMA spec):
        bump = True  when the agent's requested direction leads OFF the grid.
        Clicking a non-adjacent cell is silently rejected (no bump fired).
        This matches the original Wumpus World specification where bump
        is a percept caused by walking into a wall, not by bad pathfinding.
    """

    def __init__(self):
        self.pos:         tuple[int, int] = (0, 0)
        self.alive:       bool            = True
        self.has_gold:    bool            = False
        self.has_arrow:   bool            = True        # one arrow; used once
        self.moves:       int             = 0
        self.score:       int             = 0
        self.bump:        bool            = False       # wall-collision percept
        self.scream:      bool            = False       # True for one turn after kill
        self.won:         bool            = False
        self.death_cause: Optional[str]   = None

    def apply_move(self, world: World, target: tuple[int, int]) -> dict:
        """
        Attempt to move the agent to target cell.

        Steps:
        1. Guard: if game is over, return noop.
        2. Silently reject non-adjacent clicks (mouse UX only — no bump).
        3. Check WALL BUMP: if the requested direction goes off-grid → bump.
        4. Apply movement: update pos, increment moves, deduct score.
        5. Clear transient percept flags (bump, scream reset on every move).
        6. Check outcomes in order: death → gold → win → plain move.

        BUMP SEMANTICS (v2):
        ─────────────────────────────────────────────────────────────────────
        Classic AIMA Wumpus World defines bump as "the agent walked into a
        wall" — i.e. the requested direction leads outside the grid boundary.

        This method receives a target cell (r, c). To distinguish a
        "wall hit" from a "bad mouse click":
        - If target is adjacent (one step away) but outside [0, GRID_SIZE),
          that IS a wall hit → bump = True.
        - If target is not adjacent at all (e.g. two steps away or random
          mouse click), it is silently ignored (no bump, no cost).

        The main() loop enforces this: arrow-key moves always produce an
        adjacent-or-wall target; mouse clicks can be anywhere.
        """
        if not self.alive or self.won:
            return {"type": "noop"}

        ar, ac = self.pos
        tr, tc = target

        # ── Silent reject: target not adjacent (mouse click on far cell) ─────
        # Manhattan distance > 1 and target is not 1 step away in any direction
        row_diff = abs(tr - ar)
        col_diff = abs(tc - ac)
        is_one_step = (row_diff + col_diff == 1)   # exactly one cardinal step

        if not is_one_step:
            # Silently ignore — not a wall hit, just a bad click
            return {"type": "noop"}

        # ── Wall bump: one step in a direction but off the grid ───────────────
        # The agent "tried" to move but the wall stopped them.
        if not (0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE):
            self.bump = True
            return {"type": "bump", "pos": self.pos}

        # ── Valid move — apply it ─────────────────────────────────────────────
        self.bump   = False
        self.scream = False       # scream clears on each new action
        self.moves += 1
        self.score += SCORE_STEP  # -1 per step
        self.pos    = (tr, tc)

        # ── Check death ───────────────────────────────────────────────────────
        cause = world.is_deadly(tr, tc)
        if cause:
            self.alive       = False
            self.score      += SCORE_DEATH
            self.death_cause = cause
            return {"type": "death", "cause": cause, "pos": (tr, tc)}

        # ── Check gold pickup ─────────────────────────────────────────────────
        if not self.has_gold and (tr, tc) == world.gold:
            self.has_gold = True
            self.score   += SCORE_GOLD
            return {"type": "gold", "pos": (tr, tc)}

        # ── Check win (back at start with gold) ───────────────────────────────
        if self.has_gold and (tr, tc) == (0, 0):
            self.won    = True
            self.score += SCORE_CLIMB
            return {"type": "win", "pos": (tr, tc)}

        return {"type": "move", "pos": (tr, tc)}

    def shoot_arrow(self, world: World, direction: tuple[int, int]) -> dict:
        """
        Shoot the agent's single arrow in a cardinal direction.

        The arrow travels in a straight line from the agent's current cell
        until it either hits the Wumpus or exits the grid boundary.
        Only ONE arrow is available per game; once fired it is gone.

        Parameters:
            world     — the World instance (may be mutated if Wumpus is hit)
            direction — unit vector: (1,0)=UP, (-1,0)=DOWN, (0,1)=RIGHT,
                        (0,-1)=LEFT

        Returns an event dict with type:
            "no_arrow" — arrow already used; no action taken, no score penalty
            "kill"     — Wumpus was in the line of fire and is now dead
                         scream = True for this turn
                         world.wumpus_alive = False
            "miss"     — arrow fired but Wumpus not in trajectory
                         arrow is still consumed; score penalty still applied

        Scoring:
            SCORE_ARROW (-10) is deducted for any shot (hit or miss).
            No deduction if arrow is already gone.

        Side effects on hit:
            - world.wumpus_alive = False
            - self.scream = True
            - world is mutated in place (stench percepts will disappear
              automatically because get_percepts() checks wumpus_alive)
        """
        # ── Guard: arrow already used ─────────────────────────────────────────
        if not self.has_arrow:
            return {"type": "no_arrow"}

        # ── Consume arrow and pay cost ────────────────────────────────────────
        self.has_arrow = False
        self.scream    = False
        self.score    += SCORE_ARROW   # -10

        # ── Trace arrow path ──────────────────────────────────────────────────
        # Start one step ahead of the agent and keep going until off-grid.
        dr, dc = direction
        r, c   = self.pos[0] + dr, self.pos[1] + dc

        while 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            if world.wumpus_alive and (r, c) == world.wumpus:
                # ── HIT: Wumpus is in this cell ───────────────────────────────
                world.wumpus_alive = False
                self.scream        = True
                return {"type": "kill", "target": "wumpus", "pos": (r, c)}
            r += dr
            c += dc

        # ── MISS: arrow exited the grid without hitting anything ──────────────
        return {"type": "miss"}

    def get_percepts(self, world: World) -> dict:
        """Return current cell percepts plus bump/scream flags."""
        r, c = self.pos
        p = world.get_percepts(r, c, self.has_gold)
        p["bump"]   = self.bump
        p["scream"] = self.scream
        return p


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: KnowledgeBase
# ═══════════════════════════════════════════════════════════════════════════════
class KnowledgeBase:
    """
    The agent's internal model of what it knows about the world.

    Stores:
    - visited:          set of cells the agent has been to (and survived).
    - safe:             set of cells the agent knows are safe to enter.
    - risky:            set of cells that might contain a pit or Wumpus.
    - percepts:         dict mapping cell → {stench, breeze, glitter} observed there.
    - wumpus_loc:       inferred Wumpus location (tuple or None if unknown).
    - last_rule:        string describing the most recent inference rule that fired.
    - infer_log:        list of inference events for display in the UI log.

    NEW in v3 (AI improvement):
    - risk_scores:      dict mapping cell → float danger score (0.0 = safe, higher = more dangerous).
                        Used by AIPlayer to rank frontier cells by estimated danger.
                        Computed from: stench neighbor count × wumpus_weight +
                                       breeze neighbor count × pit_weight.
    - impossible_wumpus: set of cells PROVEN to NOT contain the Wumpus.
                        Derived from negative inference: cells adjacent to a
                        no-stench visited cell cannot be the Wumpus.
    - impossible_pit:   set of cells PROVEN to NOT contain a pit.
                        Derived from: cells adjacent to a no-breeze visited cell.
    - frontier_safe:    unvisited cells that are logically safe to enter.
    - frontier_uncertain: unvisited cells with unknown risk (not yet classifiable).
    - frontier_risky:   unvisited cells known to be dangerous.
    - ai_reasoning:     last AI decision explanation string (shown in status panel).

    The KB is SEPARATE from World — it only contains what the agent
    has logically deduced, not the ground truth.
    """

    def __init__(self):
        self.visited:           set[tuple[int, int]]  = {(0, 0)}
        self.safe:              set[tuple[int, int]]  = {(0, 0)}
        self.risky:             set[tuple[int, int]]  = set()
        self.percepts:          dict                  = {}      # (r,c) → {stench,breeze,glitter}
        self.wumpus_loc:        Optional[tuple]       = None
        self.last_rule:         str                   = "—"
        self.infer_log:         list[str]             = []
        self._seen_infer:       set[str]              = set()

        # ── NEW: negative inference sets ─────────────────────────────────────
        # Cells that have been logically RULED OUT as Wumpus / pit locations.
        # R7: neighbor of no-stench cell → cannot be Wumpus.
        # R8: neighbor of no-breeze cell → cannot be pit.
        self.impossible_wumpus: set[tuple[int, int]]  = set()
        self.impossible_pit:    set[tuple[int, int]]  = set()

        # ── NEW: probabilistic risk scores ────────────────────────────────────
        # Maps unvisited cell → composite danger score (float).
        # Higher = more dangerous. 0.0 = confirmed safe (also in kb.safe).
        #
        # Score formula (computed by InferenceEngine._rule_9_risk_scores):
        #   wumpus_contrib = count of stench-neighbors that point here × 40
        #   pit_contrib    = count of breeze-neighbors that point here  × 30
        #   impossible_bonus = -200 if cell is impossible_wumpus AND impossible_pit
        #   base            = +10 for any unvisited, non-safe, non-impossible cell
        #
        # The AI uses this to pick the LEAST risky frontier cell when forced.
        self.risk_scores:       dict[tuple, float]    = {}

        # ── NEW: classified frontier ──────────────────────────────────────────
        # Updated by InferenceEngine after every infer() call.
        # AIPlayer reads these directly — no recomputation needed per step.
        self.frontier_safe:      set[tuple[int, int]] = set()   # unvisited + confirmed safe
        self.frontier_uncertain: set[tuple[int, int]] = set()   # unvisited + unknown risk
        self.frontier_risky:     set[tuple[int, int]] = set()   # unvisited + known dangerous

        # ── NEW: AI reasoning trace ───────────────────────────────────────────
        # Human-readable explanation of why the AI chose its last action.
        # Displayed in the status panel when AI mode is active.
        self.ai_reasoning:      str                   = ""

    def record_percept(self, pos: tuple, percept: dict):
        """Save the percept observed at pos."""
        self.percepts[pos] = percept

    def visit(self, pos: tuple):
        """Mark a cell as visited (agent survived it)."""
        self.visited.add(pos)
        self.safe.add(pos)   # survived → safe

    def _log_infer(self, msg: str):
        """Add an inference event to the log (deduplicated)."""
        if msg not in self._seen_infer:
            self._seen_infer.add(msg)
            self.infer_log.insert(0, msg)
            if len(self.infer_log) > 50:
                self.infer_log.pop()

    def clear_wumpus_knowledge(self):
        """
        Called after the Wumpus is killed.

        Clears the inferred Wumpus location and removes the Wumpus cell
        from the risky set. The stench percept disappears automatically
        because World.get_percepts() checks wumpus_alive at runtime —
        no stored percept rewriting is needed.

        Also resets impossible_wumpus so re-inference can proceed cleanly
        if needed (Wumpus is dead — no more stench to reason about).
        """
        if self.wumpus_loc is not None:
            self.risky.discard(self.wumpus_loc)
            self.risk_scores.pop(self.wumpus_loc, None)
            self.wumpus_loc = None
        self.impossible_wumpus.clear()   # no longer relevant
        self.last_rule = "Wumpus killed — stench cleared"
        self._log_infer("KB updated: Wumpus dead, stench cells now safe")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: InferenceEngine  (v3 — enhanced with negative inference + risk scoring)
# ═══════════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    """
    Propositional logic inference over the KnowledgeBase.

    Implements 9 rules (6 original + 3 new for v3):

    ── ORIGINAL RULES (R1–R6) ────────────────────────────────────────────────
    R1  No stench AND no breeze at a visited cell
        → all orthogonal neighbors of that cell are SAFE.
        (If I feel nothing, nothing around me is dangerous.)

    R2  Cell (0,0) is always safe by game definition.5

    R3  Every visited cell is safe (the agent survived it).

    R4  A breeze at a visited cell
        → each unvisited, non-safe neighbor is potentially a pit (RISKY).

    R5  Stench intersection: take all stench cells, intersect their
        unvisited neighbor sets. If exactly one cell remains, that cell
        MUST be the Wumpus (logical deduction).

    R6  Any cell in both safe and risky → remove from risky.
        (Safe knowledge overrides risk suspicion.)

    ── NEW RULES (R7–R9) ─────────────────────────────────────────────────────
    R7  NEGATIVE WUMPUS INFERENCE — elimination reasoning:
        If a visited cell has NO stench → none of its unvisited neighbors
        can possibly be the Wumpus. Add them to kb.impossible_wumpus.
        Additionally: if a cell is in impossible_wumpus AND it is the only
        remaining stench-neighbor candidate, the Wumpus MUST be elsewhere.

        Corollary: if ALL unvisited neighbors of a stench cell except one
        are in impossible_wumpus, that one remaining cell IS the Wumpus
        (stronger pinpointing than pure intersection).

    R8  NEGATIVE PIT INFERENCE — elimination reasoning:
        If a visited cell has NO breeze → none of its unvisited neighbors
        can contain a pit. Add them to kb.impossible_pit.
        If a cell is in impossible_pit → remove from risky (if it was
        flagged risky only because of pit suspicion and Wumpus is ruled out).

        Corollary: a cell that is both impossible_wumpus AND impossible_pit
        is CONFIRMED SAFE and added to kb.safe.

    R9  RISK SCORE COMPUTATION — probabilistic danger assessment:
        For each unvisited, non-safe cell, compute a composite danger score:
          wumpus_risk = (# of stench-neighbor cells pointing to it) × 40
                        0 if the cell is in impossible_wumpus
          pit_risk    = (# of breeze-neighbor cells pointing to it) × 30
                        0 if the cell is in impossible_pit
          score       = wumpus_risk + pit_risk

        Store in kb.risk_scores[cell].
        Cells with score == 0 and not already safe → promoted to kb.safe.
        Cells with score > 0 → sorted into frontier_risky or frontier_uncertain.

        Frontier classification (also updated here):
          frontier_safe      = kb.safe ∩ unvisited
          frontier_uncertain = unvisited, not safe, score in (0, threshold)
          frontier_risky     = unvisited, not safe, score ≥ threshold
          threshold = 30 (one breeze-neighbor or half a stench-neighbor)
    """

    RISKY_THRESHOLD = 30   # score at or above this → frontier_risky

    @staticmethod
    def infer(kb: KnowledgeBase):
        """
        Run all 9 inference rules against kb. Mutates kb in-place.

        Order matters:
        1. R2, R3 first — establish baseline safe sets.
        2. R1 — propagate safe cells from clean-percept cells.
        3. R7, R8 — negative inference (must come after R1 adds to safe).
        4. R4 — mark risky (with impossible_pit already populated from R8).
        5. R5 — pinpoint Wumpus (benefits from impossible_wumpus from R7).
        6. R6 — resolve conflicts.
        7. R9 — compute risk scores and classify frontier (last, uses all above).
        """
        InferenceEngine._rule_2_start_safe(kb)
        InferenceEngine._rule_3_visited_safe(kb)
        InferenceEngine._rule_1_no_percepts(kb)
        InferenceEngine._rule_7_negative_wumpus(kb)
        InferenceEngine._rule_8_negative_pit(kb)
        InferenceEngine._rule_4_breeze_risky(kb)
        InferenceEngine._rule_5_wumpus_pinpoint(kb)
        InferenceEngine._rule_6_safe_not_risky(kb)
        InferenceEngine._rule_9_risk_scores(kb)

    # ═══════════════════════════════════════════════════════════════════════════
    # ORIGINAL RULES (R1–R6)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _rule_1_no_percepts(kb: KnowledgeBase):
        """
        R1: visited cell with no stench + no breeze
            → all its neighbors are safe.
        """
        for cell, p in kb.percepts.items():
            if not p.get("stench") and not p.get("breeze"):
                for nb in World.neighbors(*cell):
                    if nb not in kb.safe:
                        kb.safe.add(nb)
                        msg = f"SAFE inferred: {nb}  (no percepts at {cell})"
                        kb.last_rule = f"R1: {cell} → {nb} safe"
                        kb._log_infer(msg)

    @staticmethod
    def _rule_2_start_safe(kb: KnowledgeBase):
        """R2: (0,0) is always safe."""
        kb.safe.add((0, 0))

    @staticmethod
    def _rule_3_visited_safe(kb: KnowledgeBase):
        """R3: All visited cells are safe (agent survived)."""
        for cell in kb.visited:
            kb.safe.add(cell)

    @staticmethod
    def _rule_4_breeze_risky(kb: KnowledgeBase):
        """
        R4: Breeze at a visited cell → unvisited non-safe neighbors are risky.

        v3 improvement: skip marking a cell risky if it is in impossible_pit
        (the cell cannot contain a pit — only mark risky if Wumpus is possible).
        """
        for cell, p in kb.percepts.items():
            if p.get("breeze"):
                for nb in World.neighbors(*cell):
                    if nb not in kb.safe and nb not in kb.visited:
                        # Only mark risky if the cell is not impossible_pit,
                        # OR if Wumpus might be there (not impossible_wumpus).
                        is_impossible_pit    = nb in kb.impossible_pit
                        is_impossible_wumpus = nb in kb.impossible_wumpus
                        if not (is_impossible_pit and is_impossible_wumpus):
                            kb.risky.add(nb)

    @staticmethod
    def _rule_5_wumpus_pinpoint(kb: KnowledgeBase):
        """
        R5 (enhanced): Intersect the unvisited neighbor sets of all stench cells.
        If exactly one cell appears in all sets AND is not in impossible_wumpus,
        that cell IS the Wumpus.

        v3 improvement: also applies the corollary from R7 — if all but one
        candidate for a stench cell are in impossible_wumpus, the survivor
        is the Wumpus regardless of intersection size.
        """
        if kb.wumpus_loc is not None:
            return  # already pinpointed

        stench_cells = [c for c, p in kb.percepts.items() if p.get("stench")]
        if not stench_cells:
            return

        # For each stench cell, valid Wumpus candidates = unvisited neighbors
        # that have NOT been ruled out by R7 (impossible_wumpus).
        candidate_sets = []
        for sc in stench_cells:
            valid = {
                nb for nb in World.neighbors(*sc)
                if nb not in kb.visited
                and nb not in kb.impossible_wumpus
            }
            if not valid:
                # Contradiction: stench exists but all neighbors ruled out.
                # This shouldn't happen in a valid world; skip this cell.
                continue
            candidate_sets.append(valid)

        if not candidate_sets:
            return

        # Intersect all valid candidate sets
        intersection = candidate_sets[0]
        for s in candidate_sets[1:]:
            intersection = intersection & s

        if len(intersection) == 1:
            loc = next(iter(intersection))
            if loc != kb.wumpus_loc:   # prevent duplicate log entries
                kb.wumpus_loc = loc
                kb.last_rule  = f"R5: Wumpus pinpointed at {loc}"
                kb._log_infer(f"WUMPUS pinpointed at {loc}")
                kb.risky.add(loc)

    @staticmethod
    def _rule_6_safe_not_risky(kb: KnowledgeBase):
        """R6: Safe cells cannot be risky — remove overlap."""
        kb.risky -= kb.safe

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW RULES (R7–R9)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _rule_7_negative_wumpus(kb: KnowledgeBase):
        """
        R7: NEGATIVE WUMPUS INFERENCE (elimination reasoning).

        Premise: if a visited cell has no stench, the Wumpus is NOT in any
        of its unvisited neighbors (because stench propagates 1 step).

        This is the logical negation of the stench rule:
            ¬Stench(x) → ∀ neighbor n of x: ¬Wumpus(n)

        Additionally: if a cell is in impossible_wumpus and it is also safe
        (impossible_pit), confirm it safe.

        Corollary pinpointing:
        For each stench cell sc, compute the valid wumpus candidates
        (unvisited neighbors NOT in impossible_wumpus). If exactly ONE
        candidate remains, that cell IS the Wumpus — pin it immediately.
        This is stronger than pure intersection because it uses elimination.
        """
        # ── Part 1: negative propagation ────────────────────────────────────
        for cell, p in kb.percepts.items():
            if not p.get("stench"):
                # No stench here → no Wumpus in any neighbor
                for nb in World.neighbors(*cell):
                    if nb not in kb.visited:
                        kb.impossible_wumpus.add(nb)

        # ── Part 2: corollary pinpointing via elimination ────────────────────
        if kb.wumpus_loc is None:
            stench_cells = [c for c, p in kb.percepts.items() if p.get("stench")]
            for sc in stench_cells:
                candidates = [
                    nb for nb in World.neighbors(*sc)
                    if nb not in kb.visited
                    and nb not in kb.impossible_wumpus
                ]
                if len(candidates) == 1:
                    loc = candidates[0]
                    if kb.wumpus_loc is None:   # first pinpoint wins
                        kb.wumpus_loc = loc
                        kb.last_rule  = f"R7: Wumpus eliminated to {loc}"
                        kb._log_infer(f"WUMPUS eliminated → {loc}")
                        kb.risky.add(loc)

    @staticmethod
    def _rule_8_negative_pit(kb: KnowledgeBase):
        """
        R8: NEGATIVE PIT INFERENCE (elimination reasoning).

        Premise: if a visited cell has no breeze, none of its unvisited
        neighbors can contain a pit (because breeze propagates 1 step).

        Logical form:
            ¬Breeze(x) → ∀ neighbor n of x: ¬Pit(n)

        Corollary: a cell that is in BOTH impossible_wumpus AND impossible_pit
        cannot contain the Wumpus AND cannot contain a pit → it is SAFE.
        Add it to kb.safe immediately.

        This rule dramatically expands the safe set when the agent has
        explored clean corridors — it rules out entire rows/columns.
        """
        # ── Part 1: negative pit propagation ────────────────────────────────
        for cell, p in kb.percepts.items():
            if not p.get("breeze"):
                for nb in World.neighbors(*cell):
                    if nb not in kb.visited:
                        kb.impossible_pit.add(nb)

        # ── Part 2: double-impossible → confirmed safe ───────────────────────
        doubly_impossible = kb.impossible_wumpus & kb.impossible_pit
        for cell in doubly_impossible:
            if cell not in kb.safe and cell not in kb.visited:
                kb.safe.add(cell)
                kb.risky.discard(cell)
                kb.last_rule = f"R8: {cell} doubly impossible → SAFE"
                kb._log_infer(f"SAFE confirmed: {cell}  (not Wumpus, not pit)")

    @staticmethod
    def _rule_9_risk_scores(kb: KnowledgeBase):
        """
        R9: PROBABILISTIC RISK SCORE COMPUTATION + FRONTIER CLASSIFICATION.

        Computes a danger score for every unvisited non-safe cell, then
        classifies all unvisited cells into three frontier tiers:
          frontier_safe      — logically safe (in kb.safe), not yet visited
          frontier_uncertain — unknown risk, score below RISKY_THRESHOLD
          frontier_risky     — known dangerous, score ≥ RISKY_THRESHOLD

        Score formula for cell c (unvisited, not safe):
          wumpus_risk:
            +40 for each stench-neighbor pointing at c
            (0 for stench neighbors if c is in impossible_wumpus)
          pit_risk:
            +30 for each breeze-neighbor pointing at c
            (0 for breeze neighbors if c is in impossible_pit)
          confirmed_wumpus_bonus:
            +200 if c == kb.wumpus_loc (known Wumpus location)

        A score of 0 with no pit/wumpus possibility → promoted to safe.

        This doesn't replace logical deduction — it augments decision-making
        for cases where the AI must choose between uncertain frontier cells.
        """
        kb.risk_scores.clear()

        # Collect all unvisited cells, then restrict frontier decisions to
        # cells adjacent to something we have actually visited. Unknown cells
        # deeper in the fog may have score 0 simply because we have no evidence
        # about them yet; that is not the same thing as being safe.
        all_cells = {
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
        }
        unvisited = all_cells - kb.visited
        frontier = {
            nb
            for cell in kb.visited
            for nb in World.neighbors(*cell)
            if nb in unvisited
        }

        for cell in unvisited:
            if cell in kb.safe:
                kb.risk_scores[cell] = 0.0
                continue

            score = 0.0
            r, c  = cell

            # ── Wumpus contribution ───────────────────────────────────────────
            if cell not in kb.impossible_wumpus:
                stench_neighbors_pointing_here = sum(
                    1 for nb in World.neighbors(r, c)
                    if kb.percepts.get(nb, {}).get("stench")
                )
                score += stench_neighbors_pointing_here * 40

            # ── Pit contribution ─────────────────────────────────────────────
            if cell not in kb.impossible_pit:
                breeze_neighbors_pointing_here = sum(
                    1 for nb in World.neighbors(r, c)
                    if kb.percepts.get(nb, {}).get("breeze")
                )
                score += breeze_neighbors_pointing_here * 30

            # ── Known Wumpus penalty ─────────────────────────────────────────
            if cell == kb.wumpus_loc:
                score += 200

            # Only promote to safe when both hazards have been logically ruled
            # out. A score of 0 can also mean "no nearby evidence yet", which
            # is still unknown and should not be trusted by the AI.
            if (
                cell in kb.impossible_wumpus
                and cell in kb.impossible_pit
                and cell not in kb.safe
            ):
                kb.safe.add(cell)
                kb.risky.discard(cell)
                score = 0.0

            kb.risk_scores[cell] = score

        # ── Classify frontier ─────────────────────────────────────────────────
        kb.frontier_safe.clear()
        kb.frontier_uncertain.clear()
        kb.frontier_risky.clear()

        for cell in frontier:
            if cell in kb.safe:
                kb.frontier_safe.add(cell)
            else:
                score = kb.risk_scores.get(cell, 10.0)
                if score >= InferenceEngine.RISKY_THRESHOLD:
                    kb.frontier_risky.add(cell)
                else:
                    kb.frontier_uncertain.add(cell)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: AIPlayer  (v3 — smart safe-first navigation with risk scoring)
# ═══════════════════════════════════════════════════════════════════════════════
class AIPlayer:
    """
    Knowledge-based AI agent that navigates the Wumpus World.

    v3 improvements:
    ─────────────────────────────────────────────────────────────────────────
    1. SAFE-FIRST policy: never enters risky/uncertain cells while safe
       unexplored cells exist. The old code would fall through to frontier
       cells too eagerly.

    2. Risk-scored frontier selection: when forced to explore uncertain
       territory, picks the cell with the LOWEST risk_score, not just the
       nearest cell. Ties broken by Manhattan distance.

    3. Smart arrow usage: if kb.wumpus_loc is confirmed, agent is aligned
       horizontally or vertically, and the wumpus is in the line of fire,
       the AI shoots before moving. This clears a dangerous threat and
       often unlocks safe cells that were blocked by stench.

    4. Explicit "no safe moves" pause: the AI halts and reports its
       reasoning rather than blindly stepping into certain death.

    5. AI reasoning trace: every decision is explained in kb.ai_reasoning
       for display in the status panel.

    Priority order (strict):
    ─────────────────────────────────────────────────────────────────────────
    P1  Gold already in hand + at (0,0) → already won (handled by Game.move)
    P2  Return home safely if carrying gold (BFS through safe cells only)
    P3  Shoot Wumpus if location confirmed + agent aligned + arrow available
    P4  Explore frontier_safe (confirmed safe unvisited cells, nearest first)
    P5  Explore frontier_uncertain (lowest risk_score, then nearest)
    P6  Explore frontier_risky (absolute last resort, lowest score, nearest)
    P7  None — no move possible; AI pauses

    BFS contract: ONLY travels through kb.safe ∪ kb.visited cells.
    Risky cells are NEVER used as intermediate path steps.
    """

    # ── BFS core (unchanged from v2, but docstring updated) ──────────────────

    @staticmethod
    def bfs_path(
        start:    tuple[int, int],
        goal:     tuple[int, int],
        passable: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        BFS from start to goal through passable cells only.
        Returns the full path (excluding start) or [] if unreachable.

        v3 contract: callers always pass kb.safe | kb.visited as passable.
        Risky or unknown cells are NEVER in the passable set during normal
        navigation — only passed as `passable | {target}` for frontier steps
        where target is the ONLY risky cell allowed.
        """
        queue: deque = deque([(start, [])])
        seen:  set   = {start}

        while queue:
            pos, path = queue.popleft()
            for nb in World.neighbors(*pos):
                if nb in seen:
                    continue
                seen.add(nb)
                new_path = path + [nb]
                if nb == goal:
                    return new_path
                if nb in passable:
                    queue.append((nb, new_path))
        return []

    # ── Arrow shoot decision ──────────────────────────────────────────────────

    @staticmethod
    def _should_shoot(
        agent: Agent,
        kb:    KnowledgeBase,
        world: World
    ) -> Optional[tuple[int, int]]:
        """
        Decide whether to shoot the arrow and in which direction.

        Conditions (ALL must be true):
        1. Agent has the arrow (has_arrow = True).
        2. Wumpus location is confirmed in the KB (wumpus_loc is not None).
        3. Wumpus is alive (world.wumpus_alive).
        4. Agent is in the same ROW or COLUMN as the Wumpus.
        5. The Wumpus is in the shooting direction (i.e. the arrow would
           travel toward it, not away from it).

        Returns the direction tuple (dr, dc) to shoot, or None if
        conditions are not met.

        This method checks alignment precisely:
        - Same row, Wumpus to the RIGHT → direction (0, +1)
        - Same row, Wumpus to the LEFT  → direction (0, -1)
        - Same col, Wumpus ABOVE        → direction (+1,  0)
        - Same col, Wumpus BELOW        → direction (-1,  0)
        """
        if not isinstance(kb, KnowledgeBase):
            return None   # defensive guard
        if not agent.has_arrow:
            return None
        if kb.wumpus_loc is None:
            return None
        if not world.wumpus_alive:
            return None

        ar, ac = agent.pos
        wr, wc = kb.wumpus_loc

        if ar == wr:
            # Same row — shoot horizontally
            return (0, 1) if wc > ac else (0, -1)
        elif ac == wc:
            # Same column — shoot vertically
            return (1, 0) if wr > ar else (-1, 0)

        return None   # not aligned

    # ── Frontier selection ────────────────────────────────────────────────────

    @staticmethod
    def _best_frontier(
        pos:      tuple[int, int],
        cells:    set[tuple[int, int]],
        kb:       KnowledgeBase,
        safe_only_path: bool = False
    ) -> Optional[tuple[int, int]]:
        """
        Find the best cell from `cells` to explore next.

        Ranking:
          primary   — risk_score (ascending: lower is safer)
          secondary — Manhattan distance from current pos (ascending: closer first)

        Composite sort key: score * 100 + distance
        (Multiplying score by 100 ensures risk dominates over distance.)

        Then BFS to find the first cell in sorted order that is reachable.
        safe_only_path=True  → BFS only through safe | visited (never risky)
        safe_only_path=False → BFS through safe | visited | {target} (last resort)
        """
        if not cells:
            return None

        safe_passable = kb.safe | kb.visited

        def sort_key(cell):
            score = kb.risk_scores.get(cell, 0.0)
            dist  = abs(cell[0] - pos[0]) + abs(cell[1] - pos[1])
            return score * 100 + dist

        sorted_cells = sorted(cells, key=sort_key)

        for target in sorted_cells:
            if safe_only_path:
                path = AIPlayer.bfs_path(pos, target, safe_passable)
            else:
                # Last resort: allow stepping through one risky cell (the goal)
                path = AIPlayer.bfs_path(pos, target, safe_passable | {target})
            if path:
                return path[0]

        return None

    # ── Main decision function ────────────────────────────────────────────────

    @staticmethod
    def next_move(
        agent: Agent,
        world: World,
        kb:    KnowledgeBase
    ) -> Optional[tuple[int, int]]:
        """
        Decide the next cell for the agent to move to, or return None if stuck.

        This is called by Game.ai_step() every AI timer tick. The return value
        is the IMMEDIATE next cell (one step), not a full path.

        For each priority level, the method checks feasibility and sets
        kb.ai_reasoning with a human-readable explanation before returning.

        Returns None ONLY when all frontiers are exhausted — this causes
        Game.ai_step() to pause the AI and display "No safe moves available".
        """
        pos        = agent.pos
        safe_cells = kb.safe | kb.visited   # BFS passable set

        # ── P2: Return home if carrying gold ──────────────────────────────────
        # Highest priority after winning — BFS strictly through safe cells.
        if agent.has_gold and pos != (0, 0):
            path = AIPlayer.bfs_path(pos, (0, 0), safe_cells)
            if path:
                kb.ai_reasoning = f"Carrying gold → heading home via {path[0]}"
                return path[0]
            # No safe path home — this is a very rare edge case.
            kb.ai_reasoning = "Gold in hand but no safe path home!"
            # Fall through to other priorities to find any safe cell closer to home

        # ── P3: Shoot Wumpus if aligned ───────────────────────────────────────
        # Do this BEFORE moving so the arrow travels from a known position.
        # Game.ai_step() checks the return value from next_move() and calls
        # Game.move() — but we return a special sentinel tuple for shooting.
        # DESIGN: we signal "shoot" to ai_step via a sentinel (-1, direction).
        # This avoids adding a separate ai_should_shoot() call in Game.
        shoot_dir = AIPlayer._should_shoot(agent, world, kb)
        if shoot_dir is not None:
            kb.ai_reasoning = (
                f"Wumpus at {kb.wumpus_loc} — shooting {shoot_dir}"
            )
            # Encode as special tuple: row=-1 signals shoot action to ai_step
            return ("SHOOT", shoot_dir)

        # ── P4: Explore frontier_safe (confirmed safe unvisited cells) ────────
        # This is the normal exploration path — 100% safe moves.
        if kb.frontier_safe:
            nxt = AIPlayer._best_frontier(pos, kb.frontier_safe, kb,
                                          safe_only_path=True)
            if nxt is not None:
                kb.ai_reasoning = f"Safe exploration → {nxt}"
                return nxt

        # ── P5: Explore frontier_uncertain (below risky threshold) ────────────
        # Only entered when NO safe frontier cells exist.
        # Chooses the cell with the lowest risk score.
        if kb.frontier_uncertain:
            nxt = AIPlayer._best_frontier(pos, kb.frontier_uncertain, kb,
                                          safe_only_path=False)
            if nxt is not None:
                best_score = kb.risk_scores.get(nxt, 0.0)
                kb.ai_reasoning = (
                    f"No safe cells left — uncertain frontier {nxt} "
                    f"(risk={best_score:.0f})"
                )
                return nxt

        # ── P6: Explore frontier_risky (absolute last resort) ─────────────────
        # Only entered when safe AND uncertain frontiers are both empty.
        # Picks the lowest-scored risky cell. Will likely result in death,
        # but this is logically forced — nowhere safe to go.
        if kb.frontier_risky:
            nxt = AIPlayer._best_frontier(pos, kb.frontier_risky, kb,
                                          safe_only_path=False)
            if nxt is not None:
                best_score = kb.risk_scores.get(nxt, 0.0)
                kb.ai_reasoning = (
                    f"⚠ Last resort: risky frontier {nxt} "
                    f"(risk={best_score:.0f}) — all safe cells exhausted"
                )
                return nxt

        # ── P7: Genuinely stuck — no moves available ──────────────────────────
        kb.ai_reasoning = "No safe moves available — AI paused"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: Game
# ═══════════════════════════════════════════════════════════════════════════════
class Game:
    """
    Controller: wires World, Agent, KnowledgeBase, InferenceEngine, and
    AIPlayer together. Exposes move() and shoot() interfaces that the UI calls.

    Responsibilities:
    - Initialize and reset all components on new_game().
    - Execute moves: apply_move → sense → infer → handle event.
    - Execute arrow shots: shoot_arrow → handle event → KB update.
    - Maintain the action/event log displayed in the UI.
    - Manage AI mode: expose ai_step() that the main loop calls on a timer.
    - Track arrow flash animation state for the Renderer.

    The Game class does NOT draw anything — all rendering is in Renderer.

    NEW in v2:
    - shoot(direction) — fires the arrow; updates KB, logs, and status.
    - arrow_flash      — (start_pos, end_pos, timestamp) for the tracer animation.
    """

    def __init__(self):
        # These are set in new_game(); declared here for IDE type hints
        self.world:   World            = None
        self.agent:   Agent            = None
        self.kb:      KnowledgeBase    = None
        self.reveal:  bool             = False
        self.show_kb: bool             = False
        self.ai_mode: bool             = False

        # Log: list of (text, colour_tuple)
        self.log:    list[tuple[str, tuple]] = []
        self.status: tuple[str, tuple]       = ("", (140, 150, 168))

        # Arrow flash animation: None or (pixel_start, pixel_end, timestamp)
        # The Renderer reads this to draw a brief projectile tracer line.
        self.arrow_flash: Optional[tuple] = None
        self.FLASH_DURATION = 0.35         # seconds the tracer is visible

        self.new_game()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def new_game(self):
        """Reset everything to a fresh random game."""
        self.world      = World()
        self.agent      = Agent()
        self.kb         = KnowledgeBase()
        self.ai_mode    = False
        self.log        = []
        self.arrow_flash = None

        self._log("New game started.", (72, 82, 102))
        self._log(
            f"Wumpus@{self.world.wumpus}  "
            f"Gold@{self.world.gold}  "
            f"Pits:{len(self.world.pits)}",
            (72, 82, 102)
        )

        # Sense the start cell immediately (might be breezy / stenchy)
        self._sense_and_infer()
        self.status = ("Explore the cave. Find the gold. Return to (0,0).", (140, 150, 168))

    # ── Move interface ────────────────────────────────────────────────────────

    def move(self, target: tuple[int, int]):
        """
        Primary movement entry point.
        Called by UI on mouse click, arrow key, or AI step.
        """
        if not self.agent.alive or self.agent.won:
            return

        event = self.agent.apply_move(self.world, target)
        self._sense_and_infer()
        self._handle_event(event)

    # ── Shoot interface (NEW) ─────────────────────────────────────────────────

    def shoot(self, direction: tuple[int, int]):
        """
        Fire the agent's arrow in a cardinal direction.

        Parameters:
            direction — unit vector (dr, dc):
                        ( 1,  0) = UP
                        (-1,  0) = DOWN
                        ( 0,  1) = RIGHT
                        ( 0, -1) = LEFT

        Flow:
        1. Delegate to Agent.shoot_arrow() for game-logic outcome.
        2. Handle the returned event: update status, log, KB.
        3. Re-sense current cell (stench may have cleared after Wumpus death).
        4. Compute arrow flash animation coordinates for the Renderer.

        Called from the Pygame event loop when a shoot key is pressed.
        Not called by the AI (AI does not shoot in this version).
        """
        if not self.agent.alive or self.agent.won:
            return

        event = self.agent.shoot_arrow(self.world, direction)
        self._handle_shoot_event(event, direction)

        # Re-sense: if Wumpus died, stench at current cell must update
        self._sense_and_infer()

    def _handle_shoot_event(self, event: dict, direction: tuple[int, int]):
        """
        Translate a shoot_arrow() event dict into status, log, KB updates,
        and the arrow_flash animation tuple.

        Arrow flash tuple format: (grid_start, grid_end, timestamp)
        where grid_start and grid_end are (row, col) tuples.
        The Renderer converts these to pixel coordinates for drawing.
        """
        t = event.get("type")
        dr, dc = direction

        # Direction names for human-readable log entries
        dir_names = {(1,0): "UP", (-1,0): "DOWN", (0,1): "RIGHT", (0,-1): "LEFT"}
        dir_name  = dir_names.get(direction, "?")

        if t == "no_arrow":
            self.status = ("No arrow left!", (212, 168, 67))
            return

        # ── Compute arrow flash trajectory (from agent to grid edge) ─────────
        ar, ac   = self.agent.pos
        flash_r  = ar + dr * GRID_SIZE    # overshoot to edge
        flash_c  = ac + dc * GRID_SIZE
        # Clamp to grid boundary
        flash_r  = max(0, min(GRID_SIZE - 1, flash_r))
        flash_c  = max(0, min(GRID_SIZE - 1, flash_c))
        self.arrow_flash = (
            (ar, ac),         # start cell
            (flash_r, flash_c),  # end cell (clamped grid edge)
            time.time()       # timestamp — Renderer checks age
        )

        if t == "kill":
            # ── Wumpus killed ─────────────────────────────────────────────────
            self.status = ("🏹 Arrow fired! You killed the Wumpus! 💀", (74, 200, 120))
            self._log(f"ARROW fired {dir_name} → WUMPUS KILLED at {event['pos']}",
                      (74, 200, 120))
            # Update KB: remove Wumpus from risky/known location
            self.kb.clear_wumpus_knowledge()

        elif t == "miss":
            # ── Arrow missed ──────────────────────────────────────────────────
            self.status = (f"🏹 Arrow fired {dir_name}... it missed.", (212, 168, 67))
            self._log(f"ARROW fired {dir_name} — missed", (212, 168, 67))

    def ai_step(self):
        """
        Execute one AI decision. Called on timer when AI mode is ON.

        v3 change: next_move() may now return a ("SHOOT", direction) sentinel
        tuple instead of a cell tuple. This signals the AI wants to shoot the
        arrow before moving. We handle that here cleanly without changing the
        move() interface.

        Also updates the status panel with kb.ai_reasoning when AI is playing,
        so the player can see why the AI made its choice.
        """
        if not self.agent.alive or self.agent.won:
            self.ai_mode = False
            return

        action = AIPlayer.next_move(self.agent, self.world, self.kb)

        if action is None:
            # AI is stuck — display reasoning and pause
            reason = self.kb.ai_reasoning or "No safe moves available"
            self.status = (f"🤖 AI paused: {reason}", (212, 168, 67))
            self.ai_mode = False
            return

        # ── Handle SHOOT sentinel ─────────────────────────────────────────────
        if isinstance(action, tuple) and len(action) == 2 and action[0] == "SHOOT":
            _, direction = action
            self.shoot(direction)
            # After shooting, update status with AI reasoning
            if self.kb.ai_reasoning:
                current_msg, _ = self.status
                self.status = (f"🤖 {self.kb.ai_reasoning}", (74, 200, 120))
            return

        # ── Handle normal move ────────────────────────────────────────────────
        self.move(action)

        # Overlay AI reasoning in status if the move didn't already set a
        # more important message (death / gold / win take priority).
        msg, col = self.status
        if col == (140, 150, 168) and self.kb.ai_reasoning:
            self.status = (f"🤖 {self.kb.ai_reasoning}", (140, 150, 168))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sense_and_infer(self):
        """
        After every move or shoot (including game start):
        1. Get percepts at current cell.
        2. Record them in the KB (overwrites previous percept for this cell —
           important after Wumpus death so stench disappears).
        3. Mark current cell as visited.
        4. Run all inference rules.
        5. Flush new KB inference log entries to the UI log.
        """
        pos     = self.agent.pos
        percept = self.agent.get_percepts(self.world)
        self.kb.record_percept(pos, percept)   # overwrites → stench clears after kill
        self.kb.visit(pos)
        InferenceEngine.infer(self.kb)

        # Flush new inference entries to UI log
        for entry in self.kb.infer_log:
            if not any(entry == t for t, _ in self.log):
                self._log(entry, (74, 200, 120))

    def _handle_event(self, event: dict):
        """Translate a move event dict into a status message and log entry."""
        t = event.get("type")
        if t == "death":
            cause = "Wumpus" if event["cause"] == "wumpus" else "pit"
            self.status = (f"💀 Killed by {cause}! Game over.", (201, 64, 64))
            self._log(f"DEATH at {event['pos']} — {cause}", (201, 64, 64))
        elif t == "gold":
            self.status = ("🥇 Gold grabbed! Return to (0,0)!", (212, 168, 67))
            self._log(f"GOLD picked up at {event['pos']}", (212, 168, 67))
        elif t == "win":
            self.status = ("🏆 Escaped with the gold! You win!", (74, 200, 120))
            self._log("WIN — escaped with gold!", (74, 200, 120))
        elif t == "bump":
            # Wall bump — agent walked into grid boundary
            self.status = ("Bump! You walked into a wall.", (212, 168, 67))
            self._log(f"BUMP — wall at {event['pos']}", (212, 168, 67))
        elif t == "move":
            self.status = (
                f"Moved to {event['pos']}.  Score: {self.agent.score}",
                (140, 150, 168)
            )
            self._log(f"Move → {event['pos']}", (140, 150, 168))
        # "noop" is silently ignored

    def _log(self, msg: str, col: tuple):
        """Prepend a message to the display log (max 80 entries)."""
        self.log.insert(0, (msg, col))
        if len(self.log) > 80:
            self.log.pop()

    # ── Convenience properties for the UI ─────────────────────────────────────

    @property
    def is_over(self) -> bool:
        return not self.agent.alive or self.agent.won

    @property
    def current_percepts(self) -> dict:
        return self.kb.percepts.get(self.agent.pos, {})


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT & COLOUR CONSTANTS  (used by Renderer)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Window layout ────────────────────────────────────────────────────────────
CELL_PX    = 116    # each grid cell in pixels
GRID_PAD   = 28     # outer left padding
AXIS_W     = 26     # y-axis label column width
AXIS_H     = 28     # x-axis label row height
PANEL_GAP  = 24     # gap between grid and right panel
PANEL_W    = 370    # right panel width
PANEL_PAD  = 16     # inner panel padding
STATS_H    = 86     # bottom stats bar height
HEADER_H   = 64     # top header bar height

GRID_PIX   = CELL_PX * GRID_SIZE        # 464
GRID_AREA  = AXIS_W  + GRID_PIX         # 490

WIN_W = GRID_PAD + GRID_AREA + PANEL_GAP + PANEL_W + GRID_PAD   # ≈ 938
WIN_H = HEADER_H + GRID_PIX + AXIS_H + 20 + STATS_H             # ≈ 862

# Grid top-left corner
GX = GRID_PAD + AXIS_W
GY = HEADER_H + 12

# ─── Palette ──────────────────────────────────────────────────────────────────
BG        = ( 10,  12,  16)
BG2       = ( 17,  20,  26)
BG3       = ( 23,  27,  35)
BORDER    = ( 38,  44,  56)
BORDER2   = ( 55,  62,  78)
TEXT      = (226, 230, 238)
TEXT2     = (140, 150, 168)
TEXT3     = ( 72,  82, 102)
C_GOLD    = (212, 168,  67)
C_WUMPUS  = (201,  64,  64)
C_AGENT   = ( 74, 158, 255)
C_OK      = ( 74, 200, 120)
C_WARN    = (212, 168,  67)
C_DANGER  = (201,  64,  64)

# ─── Cell overlay tints (RGBA — drawn onto an SRCALPHA surface) ───────────────
T_STENCH  = (201,  64,  64,  48)
T_BREEZE  = ( 74, 158, 255,  38)
T_GLITTER = (212, 168,  67,  55)
T_VISITED = ( 74, 200, 120,  14)
T_WUMPUS  = (201,  64,  64,  85)
T_PIT     = ( 50,  55,  68, 150)
T_GOLD    = (212, 168,  67,  75)
T_SAFE    = ( 74, 200, 120,  26)
T_RISKY   = (200, 140,  40,  38)
T_AGENT   = ( 74, 158, 255,  38)

# ─── Percept badge colours (fg, bg) ──────────────────────────────────────────
BADGE_S = ((220,  90,  90), ( 55,  14,  14))
BADGE_B = (( 90, 170, 255), ( 12,  28,  58))
BADGE_G = ((220, 178,  80), ( 52,  40,  10))

# ─── Legend swatch colours (fill, border) ────────────────────────────────────
LEGEND_ITEMS = [
    (( 30,  70, 150), C_AGENT,  "Agent position"),
    (( 90,  22,  22), C_WUMPUS, "Wumpus 👾"),
    (( 52,  58,  70), BORDER2,  "Pit ⚫"),
    (( 90,  72,  18), C_GOLD,   "Gold 🥇"),
    (( 65,  18,  18), (140,48,48), "Stench (Wumpus adj)"),
    (( 18,  32,  64), (52,88,158), "Breeze (Pit adj)"),
    (( 18,  52,  32), (52,136,78),"Inferred safe ✓"),
    ((  8,  48,   8), (52,136,78),"Dead Wumpus ✝"),
]

# ─── Arrow flash tracer colour ────────────────────────────────────────────────
C_ARROW_FLASH = (255, 220, 80)    # bright gold tracer line colour


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_rect(
    surf:  pygame.Surface,
    rgba:  tuple,
    rect:  pygame.Rect,
    radius: int = 0
):
    """Draw a semi-transparent filled rectangle onto surf."""
    s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    if radius:
        pygame.draw.rect(s, rgba, s.get_rect(), border_radius=radius)
    else:
        s.fill(rgba)
    surf.blit(s, rect.topleft)


def rrect(
    surf:  pygame.Surface,
    color: tuple,
    rect:  pygame.Rect,
    r:     int = 7,
    w:     int = 0
):
    """Draw a rounded rectangle (filled or outline)."""
    pygame.draw.rect(surf, color, rect, border_radius=r, width=w)


def render_text(font, text: str, color: tuple) -> pygame.Surface:
    """Render a text surface."""
    return font.render(str(text), True, color)


def wrap_text(font, text: str, max_width: int) -> list[str]:
    """Split text into lines that fit within max_width pixels."""
    words = text.split()
    lines, cur = [], []
    for word in words:
        test = " ".join(cur + [word])
        if font.size(test)[0] > max_width:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: Renderer
# ═══════════════════════════════════════════════════════════════════════════════
class Renderer:
    """
    All Pygame drawing lives here. Reads Game state; never modifies it.

    Structure of one frame (draw call):
        draw(game)
         ├── _header(game)        — title bar + AI MODE button
         ├── _grid(game)          — 4x4 cell grid with axis labels
         │     └── _cell(game, r, c)  for each cell
         ├── _arrow_flash(game)   — brief tracer line after shooting (NEW)
         ├── _panel(game)         — right-side stacked panels
         │     ├── _p_status      — STATUS message
         │     ├── _p_percepts    — current cell percepts table (+ Scream)
         │     ├── _p_controls    — keyboard shortcuts reference (+ shoot keys)
         │     ├── _p_kb          — inference engine (if show_kb)
         │     ├── _p_log         — scrolling event log
         │     └── _p_legend      — colour legend
         └── _stats_bar(game)     — bottom bar: moves/visited/safe/score/ARROW

    Cell drawing order (for each cell):
        1. Base background (BG2)
        2. Percept tint overlay (stench = red, breeze = blue, glitter = gold)
        3. Reveal-all overlay (wumpus/pit/gold when V is pressed)
        4. KB overlay (safe = green, risky = amber, when K is pressed)
        5. Agent highlight tint
        6. Border (colour depends on agent/death/win state)
        7. Fog dots (if unvisited and not revealed)
        8. Entity icons (🤖 👾/✝ ⚫ 🥇)
        9. Percept badges (S / B / G — top-left corner)
       10. KB safe checkmark (bottom-left, when K is pressed)
       11. Coordinate label (bottom-right)

    NEW in v2:
        _arrow_flash() — draws a glowing tracer line for FLASH_DURATION seconds
        Dead Wumpus shown as ✝ in reveal mode
        Scream row in percepts panel highlights green on kill
        Shoot keys (I/J/L + Shift+Arrows) shown in controls panel
        ARROW stat box added to bottom stats bar
    """

    def __init__(self, screen: pygame.Surface):
        self.sc = screen
        self.F  = self._load_fonts()

        # Cache for alpha badge surfaces (minor performance optimisation)
        self._badge_cache: dict = {}

    # ── Font loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_fonts() -> dict:
        """
        Load fonts with graceful fallback to system monospace.
        JetBrains Mono is preferred but not required — the game
        will use the system monospace font if it isn't installed.
        """
        def f(name: str, size: int, bold: bool = False) -> pygame.font.Font:
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                return pygame.font.SysFont("monospace", size, bold=bold)

        return {
            "title":   f("jetbrainsmono", 17, bold=True),
            "sub":     f("jetbrainsmono", 11),
            "mono":    f("jetbrainsmono", 12),
            "mono_s":  f("jetbrainsmono", 10),
            "mono_xs": f("jetbrainsmono",  9),
            "badge":   f("jetbrainsmono", 10, bold=True),
            "section": f("jetbrainsmono", 10, bold=True),
            "coord":   f("jetbrainsmono",  9),
            "ui":      f("segoeui",       13),
            "ui_s":    f("segoeui",       11),
            "stat_v":  f("jetbrainsmono", 22, bold=True),
            "stat_l":  f("jetbrainsmono",  9),
            "emoji":   f("segoeuiemoji",  26),
            "emoji_s": f("segoeuiemoji",  20),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # MASTER DRAW
    # ═══════════════════════════════════════════════════════════════════════════

    def draw(self, game: Game):
        """Main draw call — called once per frame."""
        self.sc.fill(BG)
        self._header(game)
        self._grid(game)
        self._arrow_flash(game)    # NEW: draw tracer after grid (overlay)
        self._panel(game)
        self._stats_bar(game)
        pygame.display.flip()

    # ═══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════════

    def _header(self, game: Game):
        """
        Draw the top bar:
        - "WUMPUS  WORLD" monospace title (left)
        - "// knowledge-based agent environment" subtitle
        - "AI MODE: ON/OFF" pill button (right)
        - Horizontal divider line at the bottom
        """
        sc = self.sc
        sc.blit(render_text(self.F["title"], "WUMPUS  WORLD", TEXT), (GRID_PAD, 14))
        sc.blit(render_text(self.F["sub"],
            "// knowledge-based agent environment", TEXT3),
            (GRID_PAD + 204, 18))

        # Divider
        pygame.draw.line(sc, BORDER, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), 1)

        # AI MODE button (top-right)
        label     = f"AI MODE: {'ON ' if game.ai_mode else 'OFF'}"
        col       = C_AGENT if game.ai_mode else TEXT3
        label_s   = render_text(self.F["mono"], label, col)
        btn_w     = label_s.get_width() + 28
        btn_h     = 30
        btn_x     = WIN_W - GRID_PAD - btn_w
        btn_y     = (HEADER_H - btn_h) // 2
        btn_rect  = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        rrect(sc, BG3, btn_rect, r=5)
        rrect(sc, col, btn_rect, r=5, w=1)
        sc.blit(label_s, (btn_x + 14, btn_y + (btn_h - label_s.get_height()) // 2))

    # ═══════════════════════════════════════════════════════════════════════════
    # ARROW FLASH ANIMATION (NEW)
    # ═══════════════════════════════════════════════════════════════════════════

    def _arrow_flash(self, game: Game):
        """
        Draw a brief arrow-tracer line when the agent shoots.

        Reads game.arrow_flash = (grid_start, grid_end, timestamp).
        Converts grid (row, col) to pixel centre coordinates.
        Fades the tracer out by reducing alpha as time passes.
        Clears game.arrow_flash once the duration expires.

        Visual:
        - Bright gold line from agent cell centre toward the grid edge.
        - Secondary glow line slightly wider and more transparent.
        - Small arrowhead triangle at the tip.

        The flash lasts game.FLASH_DURATION seconds (default 0.35 s).
        """
        if game.arrow_flash is None:
            return

        start_cell, end_cell, timestamp = game.arrow_flash
        elapsed = time.time() - timestamp

        # Expire the flash after FLASH_DURATION
        if elapsed >= game.FLASH_DURATION:
            game.arrow_flash = None
            return

        # Compute alpha (255 → 0 over the duration)
        alpha = int(255 * (1.0 - elapsed / game.FLASH_DURATION))

        def cell_centre(row, col) -> tuple[int, int]:
            """Convert grid (row, col) to pixel centre."""
            px = GX + col * CELL_PX + CELL_PX // 2
            py = GY + (GRID_SIZE - 1 - row) * CELL_PX + CELL_PX // 2
            return (px, py)

        sx, sy = cell_centre(*start_cell)
        ex, ey = cell_centre(*end_cell)

        # Draw onto a temporary SRCALPHA surface so we can control alpha
        flash_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

        # Outer glow (wider, dimmer)
        glow_col = (*C_ARROW_FLASH, alpha // 3)
        pygame.draw.line(flash_surf, glow_col, (sx, sy), (ex, ey), 6)

        # Core tracer line
        core_col = (*C_ARROW_FLASH, alpha)
        pygame.draw.line(flash_surf, core_col, (sx, sy), (ex, ey), 2)

        # Arrowhead: small filled circle at the tip
        pygame.draw.circle(flash_surf, core_col, (ex, ey), 5)

        self.sc.blit(flash_surf, (0, 0))

    # ═══════════════════════════════════════════════════════════════════════════
    # GRID
    # ═══════════════════════════════════════════════════════════════════════════

    def _grid(self, game: Game):
        """Draw axis labels and all 16 cells."""
        sc = self.sc

        # Y-axis (row numbers 0–3, with row 0 at the bottom of the screen)
        for r in range(GRID_SIZE):
            sy  = GY + (GRID_SIZE - 1 - r) * CELL_PX + CELL_PX // 2
            lbl = render_text(self.F["coord"], str(r), TEXT3)
            sc.blit(lbl, (GX - AXIS_W + 4, sy - lbl.get_height() // 2))

        # X-axis (column numbers 0–3)
        for c in range(GRID_SIZE):
            sx  = GX + c * CELL_PX + CELL_PX // 2
            lbl = render_text(self.F["coord"], str(c), TEXT3)
            sc.blit(lbl, (sx - lbl.get_width() // 2, GY + GRID_PIX + 8))

        # Draw all cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._cell(game, r, c)

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL
    # ═══════════════════════════════════════════════════════════════════════════

    def _cell(self, game: Game, r: int, c: int):
        """
        Draw one grid cell at logical position (r, c).

        Screen y is inverted: row 0 (start) is at the bottom of the grid.
        Screen row = (GRID_SIZE - 1 - r) * CELL_PX + GY
        """
        sc   = self.sc
        pad  = 4
        sx   = GX + c * CELL_PX
        sy   = GY + (GRID_SIZE - 1 - r) * CELL_PX
        rect = pygame.Rect(sx + pad, sy + pad, CELL_PX - pad*2, CELL_PX - pad*2)
        cx, cy = rect.centerx, rect.centery

        world = game.world
        agent = game.agent
        kb    = game.kb

        vis       = (r, c) in kb.visited
        is_agent  = agent.pos == (r, c)
        p         = kb.percepts.get((r, c), {})
        has_s     = p.get("stench",  False)
        has_b     = p.get("breeze",  False)
        has_g     = p.get("glitter", False)
        has_pit   = (r, c) in world.pits
        has_wump  = world.wumpus_alive and (r, c) == world.wumpus
        has_gold  = not agent.has_gold and (r, c) == world.gold

        # ── 1. Base background ────────────────────────────────────────────────
        rrect(sc, BG2, rect, r=7)

        # ── 2. Percept tint (visited cells only) ─────────────────────────────
        if vis:
            if has_s and has_b:
                # Split diagonally: top = stench (red), bottom = breeze (blue)
                top = pygame.Rect(rect.x, rect.y, rect.w, rect.h // 2)
                bot = pygame.Rect(rect.x, rect.centery, rect.w, rect.h // 2)
                alpha_rect(sc, T_STENCH, top)
                alpha_rect(sc, T_BREEZE, bot)
            elif has_s:
                alpha_rect(sc, T_STENCH,  rect, radius=7)
            elif has_b:
                alpha_rect(sc, T_BREEZE,  rect, radius=7)
            elif has_g:
                alpha_rect(sc, T_GLITTER, rect, radius=7)
            else:
                alpha_rect(sc, T_VISITED, rect, radius=7)

        # ── 3. Reveal-all overlay (unvisited, V key) ─────────────────────────
        if game.reveal and not vis:
            if has_wump:
                alpha_rect(sc, T_WUMPUS, rect, radius=7)
            elif has_pit:
                alpha_rect(sc, T_PIT,    rect, radius=7)
            elif has_gold:
                alpha_rect(sc, T_GOLD,   rect, radius=7)

        # ── 4. KB inference overlay (K key) ──────────────────────────────────
        if game.show_kb and not vis:
            if (r, c) in kb.safe:
                alpha_rect(sc, T_SAFE,  rect, radius=7)
            elif (r, c) in kb.risky:
                alpha_rect(sc, T_RISKY, rect, radius=7)

        # ── 5. Agent highlight tint ───────────────────────────────────────────
        if is_agent and agent.alive:
            alpha_rect(sc, T_AGENT, rect, radius=7)

        # ── 6. Border ─────────────────────────────────────────────────────────
        if is_agent and agent.alive:
            rrect(sc, C_AGENT,  rect, r=7, w=2)
        elif not agent.alive and is_agent:
            rrect(sc, C_DANGER, rect, r=7, w=2)
        elif agent.won and (r, c) == (0, 0):
            rrect(sc, C_GOLD,   rect, r=7, w=2)
        elif vis:
            rrect(sc, BORDER2,  rect, r=7, w=1)
        else:
            rrect(sc, BORDER,   rect, r=7, w=1)

        # ── 7. Fog dots (unvisited, no reveal) ───────────────────────────────
        if not vis and not game.reveal:
            for dy in range(8, rect.h - 4, 10):
                for dx in range(8, rect.w - 4, 8):
                    pygame.draw.rect(sc, BORDER,
                        pygame.Rect(rect.x + dx, rect.y + dy, 2, 2))
        else:
            # ── 8. Entity icons ───────────────────────────────────────────────
            icons = []
            if is_agent and agent.alive:
                icons.append("🤖")
            if not agent.alive and is_agent:
                icons.append("💀")
            if game.reveal:
                if has_wump and not (is_agent and agent.alive):
                    icons.append("👾")
                # Dead Wumpus: show ✝ at its cell when reveal is ON
                if not world.wumpus_alive and (r, c) == world.wumpus:
                    icons.append("✝")
                if has_pit:
                    icons.append("⚫")
                if has_gold:
                    icons.append("🥇")
            # Dead Wumpus mark even in visited/fog-free mode (not reveal)
            if not game.reveal and vis and (not world.wumpus_alive) and (r, c) == world.wumpus:
                icons.append("✝")
            if vis and has_g and not game.reveal:
                icons.append("✨")

            if icons:
                icon_str = " ".join(icons)
                icon_surf = render_text(self.F["emoji"], icon_str, TEXT)
                sc.blit(icon_surf, icon_surf.get_rect(center=(cx, cy - 4)))

        # ── 9. Percept badges (top-left: S / B / G) ───────────────────────────
        if vis:
            bx = rect.x + 6
            by = rect.y + 6
            if has_s: bx = self._badge("S", BADGE_S, bx, by) + 3
            if has_b: bx = self._badge("B", BADGE_B, bx, by) + 3
            if has_g: self._badge("G", BADGE_G, bx, by)

        # ── 10. KB safe checkmark / risk score / frontier labels ─────────────
        if game.show_kb and not vis:
            if (r, c) in kb.safe:
                s = render_text(self.F["mono_s"], "✓", C_OK)
                sc.blit(s, (rect.x + 5, rect.bottom - s.get_height() - 4))
            elif (r, c) in kb.impossible_wumpus and (r, c) in kb.impossible_pit:
                # Doubly impossible → safe by R8 corollary
                s = render_text(self.F["mono_xs"], "✓x2", C_OK)
                sc.blit(s, (rect.x + 5, rect.bottom - s.get_height() - 4))
            else:
                # Show numeric risk score for uncertain/risky cells
                score = kb.risk_scores.get((r, c), 0.0)
                if score > 0:
                    score_col = C_DANGER if score >= 60 else C_WARN
                    rs = render_text(self.F["mono_xs"], f"r{score:.0f}", score_col)
                    sc.blit(rs, (rect.x + 4, rect.bottom - rs.get_height() - 4))
            if (r, c) == kb.wumpus_loc:
                s = render_text(self.F["mono_xs"], "W?", C_WUMPUS)
                sc.blit(s, (rect.x + 5, rect.y + 4))

        # ── 11. Coordinate label (bottom-right) ───────────────────────────────
        coord = render_text(self.F["coord"], f"{r},{c}", TEXT3)
        sc.blit(coord, (rect.right - coord.get_width() - 4,
                        rect.bottom - coord.get_height() - 3))

    def _badge(
        self,
        label:  str,
        colours: tuple,
        x:      int,
        y:      int
    ) -> int:
        """
        Draw a small coloured badge pill (e.g. 'S', 'B', 'G') and return
        the x coordinate of the right edge (for chaining multiple badges).
        """
        fg, bg = colours
        key = (label, fg, bg)
        if key not in self._badge_cache:
            s  = render_text(self.F["badge"], label, fg)
            pd = 3
            w  = s.get_width()  + pd * 2
            h  = s.get_height() + pd * 2
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill((*bg, 210))
            pygame.draw.rect(surf, (*fg, 130), surf.get_rect(),
                             border_radius=3, width=1)
            surf.blit(s, (pd, pd))
            self._badge_cache[key] = (surf, s)
        surf, s = self._badge_cache[key]
        self.sc.blit(surf, (x, y))
        return x + surf.get_width()

    # ═══════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL
    # ═══════════════════════════════════════════════════════════════════════════

    def _panel(self, game: Game):
        """
        Draw all right-side panels in vertical stack order.
        Each _p_* method returns the bottom y of the box it drew,
        so the next box starts right below it.
        """
        px = GRID_PAD + GRID_AREA + PANEL_GAP
        py = HEADER_H + 12

        py = self._p_status(game,   px, py) + 10
        py = self._p_percepts(game, px, py) + 10
        py = self._p_controls(game, px, py) + 10
        if game.show_kb:
            py = self._p_kb(game,   px, py) + 10
        py = self._p_log(game,      px, py) + 10
        self._p_legend(game,        px, py)

    def _box(self, py: int, h: int) -> pygame.Rect:
        """Draw a standard panel box and return its Rect."""
        px   = GRID_PAD + GRID_AREA + PANEL_GAP
        rect = pygame.Rect(px, py, PANEL_W - GRID_PAD, h)
        rrect(self.sc, BG2,    rect, r=8)
        rrect(self.sc, BORDER, rect, r=8, w=1)
        return rect

    def _section_label(self, text: str, x: int, y: int) -> int:
        """Draw a section title in small caps and return y below it."""
        s = render_text(self.F["section"], text, TEXT3)
        self.sc.blit(s, (x, y))
        return y + s.get_height() + 7

    # ── STATUS panel ──────────────────────────────────────────────────────────

    def _p_status(self, game: Game, px: int, py: int) -> int:
        rect = self._box(py, 84)
        cy   = self._section_label("STATUS", rect.x + PANEL_PAD, rect.y + 10)
        msg, col = game.status
        for line in wrap_text(self.F["ui"], msg, rect.w - PANEL_PAD * 2)[:2]:
            s = render_text(self.F["ui"], line, col)
            self.sc.blit(s, (rect.x + PANEL_PAD, cy))
            cy += s.get_height() + 2
        return rect.bottom

    # ── PERCEPTS panel ────────────────────────────────────────────────────────

    def _p_percepts(self, game: Game, px: int, py: int) -> int:
        """
        Show the 5 percepts for the current cell.
        Each row: emoji icon | name | YES/— value (right-aligned).
        """
        rect  = self._box(py, 182)
        cy    = self._section_label(
            "PERCEPTS — CURRENT CELL", rect.x + PANEL_PAD, rect.y + 10)
        p     = game.current_percepts
        rows  = [
            ("💀", "Stench",  p.get("stench",  False), C_WUMPUS),
            ("💨", "Breeze",  p.get("breeze",  False), C_AGENT),
            ("✨", "Glitter", p.get("glitter", False), C_GOLD),
            ("🚧", "Bump",    game.agent.bump,           C_WARN),
            ("😱", "Scream",  game.agent.scream,          C_OK),
        ]
        row_h = (rect.bottom - cy - 4) // len(rows)
        for icon, name, active, acol in rows:
            icon_s = render_text(self.F["emoji_s"], icon, TEXT2)
            name_s = render_text(self.F["ui"],      name, TEXT2)
            val_s  = render_text(
                self.F["mono"],
                "YES" if active else "—",
                acol  if active else TEXT3
            )
            self.sc.blit(icon_s, (rect.x + PANEL_PAD,      cy + 1))
            self.sc.blit(name_s, (rect.x + PANEL_PAD + 28, cy + 3))
            self.sc.blit(val_s,  (rect.right - val_s.get_width() - PANEL_PAD, cy + 3))
            pygame.draw.line(self.sc, BORDER,
                (rect.x + PANEL_PAD,     cy + row_h - 1),
                (rect.right - PANEL_PAD, cy + row_h - 1), 1)
            cy += row_h
        return rect.bottom

    # ── CONTROLS panel ────────────────────────────────────────────────────────

    def _p_controls(self, game: Game, px: int, py: int) -> int:
        """
        Display keyboard shortcut reference.
        Each row: [KEY] pill | description.
        Active toggles (reveal, show_kb, ai_mode) are highlighted in blue.
        Shoot keys shown in gold (arrow colour) at the bottom.
        """
        rect  = self._box(py, 198)   # taller to fit shoot key rows
        cy    = self._section_label("CONTROLS", rect.x + PANEL_PAD, rect.y + 10)
        items = [
            ("[R]",      "New game",
             False,              TEXT2,         C_AGENT),
            ("[V]",      f"Show All: {'ON' if game.reveal   else 'OFF'}",
             game.reveal,        C_AGENT,       C_AGENT),
            ("[K]",      f"Inference: {'ON' if game.show_kb else 'OFF'}",
             game.show_kb,       C_AGENT,       C_AGENT),
            ("[A/Spc]",  f"AI Auto-play: {'ON' if game.ai_mode else 'OFF'}",
             game.ai_mode,       C_AGENT,       C_AGENT),
            # ── Shoot keys (I J L Shift+Down) ────────────────────────────────
            ("[I]",      "Shoot UP",    False,  C_ARROW_FLASH, C_ARROW_FLASH),
            ("[J]",      "Shoot LEFT",  False,  C_ARROW_FLASH, C_ARROW_FLASH),
            ("[L]",      "Shoot RIGHT", False,  C_ARROW_FLASH, C_ARROW_FLASH),
            ("[Sh+↓]",   "Shoot DOWN",  False,  C_ARROW_FLASH, C_ARROW_FLASH),
        ]
        for key_lbl, desc, active, key_col, desc_col in items:
            ks  = render_text(self.F["mono"], key_lbl, key_col)
            kw  = ks.get_width() + 10
            kh  = ks.get_height() + 5
            kr  = pygame.Rect(rect.x + PANEL_PAD, cy, kw, kh)
            rrect(self.sc, BG3,     kr, r=3)
            rrect(self.sc, BORDER2, kr, r=3, w=1)
            self.sc.blit(ks, (kr.x + 5, kr.y + 2))
            ds = render_text(self.F["ui_s"], desc, desc_col if active else TEXT2)
            self.sc.blit(ds, (kr.right + 8, cy + 3))
            cy += kh + 5
        return rect.bottom

    # ── INFERENCE ENGINE panel ────────────────────────────────────────────────

    def _p_kb(self, game: Game, px: int, py: int) -> int:
        """
        Show KB state when inference overlay is enabled (K key).

        v3 additions:
        - Frontier tier counts (safe / uncertain / risky)
        - Top risk scores for uncertain cells
        - impossible_wumpus / impossible_pit set sizes
        - AI reasoning trace (last decision explanation)
        """
        rect  = self._box(py, 200)   # taller to fit new v3 rows
        cy    = self._section_label("INFERENCE ENGINE", rect.x + PANEL_PAD, rect.y + 10)
        kb    = game.kb

        # ── Compute display values ────────────────────────────────────────────
        safe_unv   = [c for c in kb.safe if c not in kb.visited]
        risky_lst  = list(kb.risky)[:4]
        wloc       = str(kb.wumpus_loc) if kb.wumpus_loc else "unknown"
        n_imp_w    = len(kb.impossible_wumpus)
        n_imp_p    = len(kb.impossible_pit)

        # Top 3 uncertain cells by risk score (ascending)
        uncertain_sorted = sorted(
            kb.frontier_uncertain,
            key=lambda c: kb.risk_scores.get(c, 0.0)
        )[:3]
        uncertain_str = (
            "  ".join(f"{c}:{kb.risk_scores.get(c,0):.0f}" for c in uncertain_sorted)
            if uncertain_sorted else "none"
        )

        # Frontier tier counts
        frontier_str = (
            f"✓{len(kb.frontier_safe)} "
            f"?{len(kb.frontier_uncertain)} "
            f"⚠{len(kb.frontier_risky)}"
        )

        # AI reasoning (truncated to fit)
        ai_rsn = kb.ai_reasoning[:38] if kb.ai_reasoning else "—"

        rows = [
            ("Safe (unvis):", str(safe_unv[:3]) if safe_unv else "none", C_OK),
            ("Risky cells:",  str(risky_lst)     if risky_lst else "none", C_WARN),
            ("Wumpus at:",    wloc,                                        C_WUMPUS),
            ("Frontier:",     frontier_str,                                C_AGENT),
            ("Uncert/score:", uncertain_str,                               C_WARN),
            ("¬Wumpus cnt:",  str(n_imp_w),                               TEXT2),
            ("¬Pit cnt:",     str(n_imp_p),                               TEXT2),
            ("Last rule:",    kb.last_rule[:36],                          TEXT2),
            ("AI reason:",    ai_rsn,                                      C_OK),
        ]
        for label, val, col in rows:
            ls = render_text(self.F["mono_s"], label, TEXT3)
            vs = render_text(self.F["mono_xs"], val,  col)
            self.sc.blit(ls, (rect.x + PANEL_PAD, cy))
            self.sc.blit(vs, (rect.x + PANEL_PAD, cy + ls.get_height() + 1))
            cy += ls.get_height() + vs.get_height() + 3
            if cy > rect.bottom - 4:
                break
        return rect.bottom

    # ── EVENT LOG panel ───────────────────────────────────────────────────────

    def _p_log(self, game: Game, px: int, py: int) -> int:
        """
        Scrolling event log showing the most recent moves and inference events.
        Auto-sizes to available vertical space above the legend.
        """
        legend_h = 186
        avail    = WIN_H - STATS_H - py - 10 - legend_h - 10
        h        = max(min(avail, 220), 90)
        rect     = self._box(py, h)
        cy       = self._section_label("EVENT LOG", rect.x + PANEL_PAD, rect.y + 10)
        line_h   = 14
        max_ln   = (rect.bottom - cy - 6) // line_h
        for msg, col in game.log[:max_ln]:
            s = render_text(self.F["mono_xs"], msg[:48], col)
            self.sc.blit(s, (rect.x + PANEL_PAD, cy))
            cy += line_h
            if cy + line_h > rect.bottom - 4:
                break
        return rect.bottom

    # ── LEGEND panel ──────────────────────────────────────────────────────────

    def _p_legend(self, game: Game, px: int, py: int):
        """
        Static legend showing what each cell colour/icon means.
        Only drawn if it fits above the stats bar.
        """
        h = 186
        if py + h > WIN_H - STATS_H - 6:
            return
        rect = self._box(py, h)
        cy   = self._section_label("LEGEND", rect.x + PANEL_PAD, rect.y + 10)
        for (bg, border, label) in LEGEND_ITEMS:
            sw = pygame.Rect(rect.x + PANEL_PAD, cy + 2, 14, 14)
            pygame.draw.rect(self.sc, bg,     sw, border_radius=3)
            pygame.draw.rect(self.sc, border, sw, border_radius=3, width=1)
            ls = render_text(self.F["ui_s"], label, TEXT2)
            self.sc.blit(ls, (sw.right + 8, cy + 1))
            cy += 24

    # ═══════════════════════════════════════════════════════════════════════════
    # STATS BAR (bottom)
    # ═══════════════════════════════════════════════════════════════════════════

    def _stats_bar(self, game: Game):
        """
        Draw the bottom stats bar with five boxes:
        MOVES | VISITED | INFERRED SAFE | SCORE | ARROW

        ARROW box shows:
          "✓ READY"  — arrow available (green)
          "✗ USED"   — arrow fired (dim red)

        Spans the full bottom of the window.
        """
        bar_y = WIN_H - STATS_H
        pygame.draw.line(self.sc, BORDER, (0, bar_y), (WIN_W, bar_y), 1)

        # Arrow status value and colour
        arrow_val = "✓ READY" if game.agent.has_arrow else "✗ USED"
        arrow_col = C_OK      if game.agent.has_arrow else C_DANGER

        stats = [
            ("MOVES",         str(game.agent.moves),                    TEXT),
            ("VISITED",       str(len(game.kb.visited)),                 TEXT),
            ("INFERRED SAFE", str(len([c for c in game.kb.safe
                                       if c not in game.kb.visited])),   TEXT),
            ("SCORE",         str(game.agent.score),                    TEXT),
            ("ARROW",         arrow_val,                                 arrow_col),
        ]

        # Distribute all 5 boxes evenly across full window width (minus padding)
        total_w = WIN_W - GRID_PAD * 2
        slot_w  = total_w // len(stats)

        for i, (label, val, val_col) in enumerate(stats):
            bx  = GRID_PAD + i * slot_w
            box = pygame.Rect(bx + 6, bar_y + 10, slot_w - 12, STATS_H - 20)
            rrect(self.sc, BG2,    box, r=6)
            rrect(self.sc, BORDER, box, r=6, w=1)

            vs = render_text(self.F["stat_v"], val,   val_col)
            ls = render_text(self.F["stat_l"], label, TEXT3)

            # Value: scale font size down if text is too wide
            if vs.get_width() > box.w - 8:
                vs = render_text(self.F["mono_s"], val, val_col)

            self.sc.blit(vs, vs.get_rect(centerx=box.centerx,
                                          centery=box.centery - 7))
            self.sc.blit(ls, ls.get_rect(centerx=box.centerx,
                                          bottom=box.bottom - 5))


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def mouse_to_cell(mx: int, my: int) -> Optional[tuple[int, int]]:
    """
    Convert a mouse pixel position to a grid (row, col) coordinate.
    Returns None if the click is outside the grid area.

    Screen layout: GX = left edge of column 0, GY = top edge of row 3.
    Row 0 is at the BOTTOM of the grid (matching the JS version).
    """
    rx = mx - GX
    ry = my - GY
    if rx < 0 or ry < 0 or rx >= GRID_PIX or ry >= GRID_PIX:
        return None
    c = rx // CELL_PX
    r = (GRID_SIZE - 1) - ry // CELL_PX    # invert y so row 0 = bottom
    return (r, c) if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE else None


# Arrow / WASD key → (delta_row, delta_col)
ARROW_DELTA = {
    pygame.K_UP:    ( 1,  0),   pygame.K_w: ( 1,  0),
    pygame.K_DOWN:  (-1,  0),   pygame.K_s: (-1,  0),
    pygame.K_LEFT:  ( 0, -1),   pygame.K_a: ( 0, -1),
    pygame.K_RIGHT: ( 0,  1),   pygame.K_d: ( 0,  1),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Pygame event loop.

    Each iteration:
    1. Tick the clock (cap at FPS).
    2. Process all pending events (quit, keyboard, mouse click).
    3. If AI mode is active and enough time has passed, call game.ai_step().
    4. Call renderer.draw(game) to render the current frame.

    The game state is stored in `game` (Game instance).
    The renderer reads game state but never modifies it.

    ── Shoot key mapping (NEW) ────────────────────────────────────────────────
    I               → Shoot UP    direction ( 1,  0)
    J               → Shoot LEFT  direction ( 0, -1)
    L               → Shoot RIGHT direction ( 0,  1)
    Shift + Down    → Shoot DOWN  direction (-1,  0)

    Rationale for key choices:
    - I/J/L form a natural "shoot" cluster on QWERTY (like vim hjkl).
    - Shift+Down for shoot-down avoids conflict with [K] = toggle inference.
    - None conflict with movement (arrow keys / WASD) or any existing hotkey.

    ── Bump movement (UPDATED) ────────────────────────────────────────────────
    Arrow keys / WASD always attempt a one-step move in a cardinal direction.
    If that step is outside the grid, the target will be out of bounds.
    Agent.apply_move() detects this (out-of-bounds one-step) → sets bump=True.
    Mouse clicks on non-adjacent cells are silently ignored (no bump).
    """
    pygame.init()
    pygame.display.set_caption("Wumpus World")

    screen   = pygame.display.set_mode((WIN_W, WIN_H))
    clock    = pygame.time.Clock()
    renderer = Renderer(screen)
    game     = Game()
    last_ai  = time.time()

    # ── Shoot direction lookup ──────────────────────────────────────────────
    # Maps keyboard key → (delta_row, delta_col) for arrow direction.
    # NOTE: Shift+Down is checked via mods bitmask, not a separate key entry.
    SHOOT_KEYS = {
        pygame.K_i: ( 1,  0),   # I  → shoot UP
        pygame.K_j: ( 0, -1),   # J  → shoot LEFT
        pygame.K_l: ( 0,  1),   # L  → shoot RIGHT
        # Shift+Down handled separately below (K_DOWN + KMOD_SHIFT)
    }

    while True:
        clock.tick(FPS)

        # ── Event handling ─────────────────────────────────────────────────
        for event in pygame.event.get():

            # Window close button
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Keyboard
            elif event.type == pygame.KEYDOWN:
                k    = event.key
                mods = pygame.key.get_mods()

                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

                elif k == pygame.K_r:
                    # New game — reset everything
                    game.new_game()

                elif k == pygame.K_v:
                    # Toggle reveal-all (show Wumpus, pits, gold positions)
                    game.reveal = not game.reveal

                elif k == pygame.K_k and not (mods & pygame.KMOD_SHIFT):
                    # Toggle inference overlay (K without Shift)
                    game.show_kb = not game.show_kb

                elif k == pygame.K_a:
                    # Toggle AI auto-play
                    game.ai_mode = not game.ai_mode
                    last_ai = time.time()

                elif k == pygame.K_SPACE:
                    # Single AI step
                    game.ai_step()

                # ── Shoot keys ─────────────────────────────────────────────
                elif k in SHOOT_KEYS:
                    # I, J, L → shoot in their respective directions
                    game.shoot(SHOOT_KEYS[k])

                elif k == pygame.K_DOWN and (mods & pygame.KMOD_SHIFT):
                    # Shift+Down → shoot DOWN
                    game.shoot((-1, 0))

                # ── Movement keys ──────────────────────────────────────────
                elif k in ARROW_DELTA:
                    # Arrow key or WASD: compute target (may be out-of-bounds)
                    # apply_move() will set bump=True if it is.
                    dr, dc = ARROW_DELTA[k]
                    nr     = game.agent.pos[0] + dr
                    nc     = game.agent.pos[1] + dc
                    # Pass the out-of-bounds target; apply_move handles it.
                    game.move((nr, nc))

            # Mouse click — silently ignored if non-adjacent (no bump)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cell = mouse_to_cell(*event.pos)
                if cell:
                    game.move(cell)

        # ── AI auto-step (timer-driven) ─────────────────────────────────────
        if game.ai_mode and not game.is_over:
            if time.time() - last_ai >= AI_SPEED:
                game.ai_step()
                last_ai = time.time()

        # ── Render ─────────────────────────────────────────────────────────
        renderer.draw(game)

    pygame.quit()


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
