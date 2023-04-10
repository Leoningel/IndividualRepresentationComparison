from __future__ import annotations

from abc import ABC
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from geneticengine.core.grammar import extract_grammar
from geneticengine.metahandlers.lists import ListSizeBetween

from examples.utils.wrapper import run_experiments


# This example is based on the grammar given in https://link.springer.com/chapter/10.1007/978-3-319-55696-3_20

map = """.###............................
...#............................
...#.....................###....
...#....................#....#..
...#....................#....#..
...####.#####........##.........
............#................#..
............#.......#...........
............#.......#........#..
............#.......#...........
....................#...........
............#................#..
............#...................
............#.......#.....###...
............#.......#..#........
.................#..............
................................
............#...........#.......
............#...#..........#....
............#...#...............
............#...#...............
............#...#.........#.....
............#..........#........
............#...................
...##. .#####....#...............
.#..............#...............
.#..............#...............
.#......#######.................
.#.....#........................
.......#........................
..####..........................
................................"""


class ActionMain(ABC):
    pass


class Action(ABC):
    pass


@dataclass
class ActionBlock(ActionMain):
    actions: Annotated[list[Action], ListSizeBetween(2, 30)]


@dataclass
class IfFood(Action):
    yes_food: Action
    no_food: Action


@dataclass
class Move(Action):
    pass


@dataclass
class Right(Action):
    pass


@dataclass
class Left(Action):
    pass


class Direction(Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


class Position(Enum):
    EMPTY = 0
    FOOD = 1


def map_from_string(map_str: str) -> list[list[Position]]:
    return [[pos == "#" and Position.FOOD or Position.EMPTY for pos in line] for line in map_str.split("\n")]


def next_pos(
    pos: tuple[int, int, Direction],
    map: list[list[Position]],
) -> tuple[int, int]:
    masks = {
        Direction.EAST: (0, 1),
        Direction.SOUTH: (1, 0),
        Direction.WEST: (0, -1),
        Direction.NORTH: (-1, 0),
    }
    row = (pos[0] + masks[pos[2]][0]) % len(map)
    col = (pos[1] + masks[pos[2]][1]) % len(map[0])
    return (row, col)


def food_in_front(pos: tuple[int, int, Direction], map: list[list[Position]]) -> bool:
    (row, col) = next_pos(pos, map)
    return map[row][col] == Position.FOOD


def simulate(a: Action, map_str: str) -> int:
    next_instructions: list[Action] = [a]
    food_consumed = 0
    map = map_from_string(map_str)
    current_pos: tuple[int, int, Direction] = (
        0,
        0,
        Direction.EAST,
    )  # row, col, direction
    while next_instructions:
        current_instruction = next_instructions.pop(0)  # Default is -1
        if isinstance(
            current_instruction,
            ActionBlock,
        ):  # ActionBlock contains list of action lists.
            for action in reversed(current_instruction.actions):
                next_instructions = [action] + next_instructions
        elif isinstance(current_instruction, IfFood):
            if food_in_front(current_pos, map):
                next_instructions.insert(0, current_instruction.yes_food)
            else:
                next_instructions.insert(0, current_instruction.no_food)
        elif isinstance(current_instruction, Move):
            (row, col) = next_pos(current_pos, map)
            if map[row][col] == Position.FOOD:
                food_consumed += 1
                map[row][col] = Position.EMPTY
            current_pos = (row, col, current_pos[2])
        elif isinstance(current_instruction, Left):
            current_pos = (
                current_pos[0],
                current_pos[1],
                Direction((current_pos[2].value - 1) % 4),
            )
        elif isinstance(current_instruction, Right):
            current_pos = (
                current_pos[0],
                current_pos[1],
                Direction((current_pos[2].value + 1) % 4),
            )
    return food_consumed


grammar = extract_grammar([ActionBlock, Action, IfFood, Move, Right, Left], ActionMain)


def fitness_function(p):
    return simulate(p, map)


params = {
    "MINIMIZE": False,
    "NUMBER_OF_ITERATIONS": 100,
    "MIN_INIT_DEPTH": 2,
    "MIN_DEPTH": None,
    "MAX_INIT_DEPTH": 6,
    "MAX_DEPTH": 10,
    "POPULATION_SIZE": 20,
    "ELITISM": 1,
    "TARGET_FITNESS": None,
}

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-r", "--representation", dest="representation", type=int, default=0)
    args = parser.parse_args()

    run_experiments(
        grammar,
        ff=fitness_function,
        ff_test=None,
        folder_name="santafe",
        seed=args.seed,
        params=params,
        repr_code=args.representation,
    )
