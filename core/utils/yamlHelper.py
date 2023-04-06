from __future__ import annotations  # nopep8

from enum import Enum

import os
import typing


class PAIR_MAP_COUNTER_STATE(Enum):
    """
    Its value determine the next state
    we start with counter
    """
    COUNTER = 'COUNT'
    COUNT = 'PAIR'
    PAIR = 'LEFT_PAIR'
    LEFT_PAIR = 'RIGHT_PAIR'
    RIGHT_PAIR = 'COUNT'


class YamlHelper:

    @classmethod
    def load_central2context(cls, path: str):
        """
        Load the yaml of central2context, the normal yaml use too much memory and too slow
        """
        central2context: dict[str, set[str]] = {}
        with open(path, 'r') as fileHandler:
            current_central = ''
            current_context = []
            for line in fileHandler:
                line = line.strip().strip("""\"\'""")
                if line[-1] == ':':
                    if len(current_context) > 0:
                        central2context[current_central] = set(current_context)
                    current_central = line[:-1]
                    current_context = []
                elif line.startswith('- '):
                    current_context.append(line[2:])
            else:  # handle last central
                if len(current_context) > 0:
                    central2context[current_central] = set(current_context)
        return central2context

    @classmethod
    def load_pair_map_counter(cls, path: str):
        """
        Load the yaml of pair map counter, the normal yaml use too much memory and too slow
        """
        pair_map_counter: dict[typing.Tuple[str, str], int] = {}
        with open(path, 'r') as fileHandler:
            current_loader_state = PAIR_MAP_COUNTER_STATE.COUNTER
            current_count = 0
            left_pair = ''
            right_pair = ''
            for line in fileHandler:
                line = line.strip().strip("""\"\'""")
                if current_loader_state == PAIR_MAP_COUNTER_STATE.COUNT:
                    current_count = int(line.rsplit(maxsplit=1)[-1])
                elif current_loader_state == PAIR_MAP_COUNTER_STATE.LEFT_PAIR:
                    left_pair = line.rsplit(maxsplit=1)[-1]
                elif current_loader_state == PAIR_MAP_COUNTER_STATE.RIGHT_PAIR:
                    right_pair = line.rsplit(maxsplit=1)[-1]
                    # This stage we have to append to the dict
                    pair_map_counter[(left_pair, right_pair,)] = current_count
                # go to next state next line
                current_loader_state = PAIR_MAP_COUNTER_STATE[current_loader_state.value]
        return pair_map_counter
