from game import Game, State

class Kalah(Game):
    def __init__(self, p, s=4):
        ''' Creates a Kalah board with the given number of houses
            per side (plus two store pits) each containing the given.
            p -- a nonnegative integer
        '''
        if p < 0:
            raise ValueError('Number of pits must be positive: %d' % p)

        if s < 0:
            raise ValueError('Number of seeds must be positive: %d' % s)

        # pits are numbered 0, ..., 2p-1 with p and 2p+1 being the home
        # pit for player 0 and player 1 respectively and pits numbered
        # clockwise around the board
        self.pits = p;
        self._start_seeds = s
        self.size = 2 * p + 2
        self.stores = [p, 2 * p + 1]

        # the sequence to sow in starting from each house
        self.sequence = []
        # sequences for player 0
        for i in range(0, p):
            self.sequence.append([])
            for j in range(1, self.size + 1):
                if (i + j) % self.size != self.stores[1]:
                    self.sequence[-1].append((i + j) % self.size)
        self.sequence.append(None) # no sequence for P0's store

        # sequences for player 1
        for i in range(p + 1, self.size - 1):
            self.sequence.append([])
            for j in range(1, self.size + 1):
                if (i + j) % self.size != self.stores[0]:
                    self.sequence[-1].append((i + j) % self.size)
        self.sequence.append(None) # no sequence for P1's store

        # compute which pits are opposite each other (home pits have no opposite)
        # and which pits are owned (sowable from) by which player
        self.opposite = [None] * (self.size)
        self.owner = [None] * (self.size)
        for i in range(0, p):
            self.opposite[i] = self.size - 2 - i
            self.opposite[self.size - 2 - i] = i
            self.owner[i] = 0
            self.owner[p + 1 + i] = 1

            
    def initial_state(self):
        ''' Creates the initial state for this board.
        '''
        seeds = []
        for i in range(0, self.size):
            if self.owner[i] is not None:
                seeds.append(self._start_seeds)
            else:
                seeds.append(0)
        return Kalah.State(self, seeds, 0)

    
    class State(State):
        def __init__(self, board, seeds, turn):
            if board is None:
                raise ValueError('board cannot be None')
            if len(seeds) != board.size:
                raise ValueError('mismatch between size of seeds list and size of board: %d vs %d' % len(seeds), board.size)
            if turn != 0 and turn != 1:
                raise ValueError('invalid turn %d' % turn)
            
            self._board = board
            self._seeds = seeds
            self._turn = turn

            self._seeds_left = [sum(seeds[0:board.pits]), sum(seeds[board.pits + 1:board.pits * 2 + 1])]
            
            self._compute_hash()

            
        def is_initial(self):
            ''' Determines if this state is the initial state.
            '''
            return (self._seeds[self._board.stores[0]] == 0
                    and self._seeds[self._board.stores[1]] == 0
                    and self._seeds[0] == self._seeds_left[0] / self._board.pits)

            
        def actor(self):
            ''' Returns the index of the player who makes the next move from
                this state.  The index will be 0 or 1.
            '''
            return self._turn

        
        def is_legal(self, p):
            ''' Determines if sowing from the given move is legal from this state.

                p -- the index of a pit in this state
            '''
            if p < 0 or p >= self._board.size:
                raise ValueError('Illegal house %d' % p)

            return self._board.owner[p] == self._turn and self._seeds[p] > 0

        
        def get_actions(self):
            ''' Returns a list of legal moves from this state.
                The list of moves is given as a list of pits to sow from.
                Pits are indexed clockwise starting with 0 for player 0's
                first pit.
            '''
            moves = []
            if self._turn == 0:
                first = 0
            else:
                first = self._board.pits + 1

            for i in range(0, self._board.pits):
                if self._seeds[first + i] > 0:
                    moves.append(first + i)

            return sorted(moves, reverse=True)

        
        def is_capture(self, pit):
            ''' Determines if the given move from this state is a
                capturing move.

                pit -- the index of a legal pit to sow from in this state
            '''
            if (pit < 0 or pit >= self._board.size) or self._board.owner[pit] != self._turn or self._seeds[pit] <= 0:
                raise ValueError('Illegal move: %d' % pit)

            _, _, _, last = self._moving(pit)
            
            return self._board.owner[last] == self._turn and self._seeds[last] == 0 and self._board.opposite[last] is not None and self._seeds[self._board.opposite[last]] > 0

        
        def is_move_again(self, pit):
            ''' Determines if the given move from this state results
                in a free move.

                pit -- the index of a legal pit to sow from in this state
            '''
            if (pit < 0 or pit >= self._board.size) or self._board.owner[pit] != self._turn or self._seeds[pit] <= 0:
                raise ValueError('Illegal move: %d' % pit)

            _, _, _, last = self._moving(pit)
            
            return last == self._board.stores[self._turn]


        def _moving(self, pit):
            # number of seeds sown
            sowing = self._seeds[pit];
            
            # number of complete times around
            timesAround = sowing // (self._board.size - 1)

            # number of pits to sow into after complete
            extras = sowing % (self._board.size - 1)

            # pit ending in
            last = self._board.sequence[pit][(extras - 1 + self._board.size - 1) % (self._board.size - 1)]

            return (sowing, timesAround, extras, last)

            
        def successor(self, p):
            ''' Returns the state that results from sowing from the given pit
                from this state.

                p -- the index of a legal pit to sow from in this state
            '''
            if (p < 0 or p >= self._board.size) or self._board.owner[p] != self._turn or self._seeds[p] <= 0:
                raise ValueError('Illegal move: %d' % p)

            succ = Kalah.State(self._board, self._seeds[:], self._turn)

            sowing, timesAround, extras, last = self._moving(p)
            
            succ._seeds[p] = 0
            succ._seeds_left[self._turn] -= sowing

            for i in range(0, extras):
                pit = succ._board.sequence[p][i]
                succ._seeds[pit] += timesAround + 1
                if succ._board.owner[pit] is not None:
                    succ._seeds_left[succ._board.owner[pit]] += timesAround + 1
            if timesAround > 0:
                for i in range(extras, self._board.size - 1):
                    pit = succ._board.sequence[p][i]
                    succ._seeds[pit] += timesAround
                    if succ._board.owner[pit] is not None:
                        succ._seeds_left[succ._board.owner[pit]] += timesAround

            # capture opposite seeds if end in own empty pit and opposite is not empty
            if succ._seeds[last] == 1 and succ._board.opposite[last] is not None and succ._seeds[succ._board.opposite[last]] > 0 and succ._board.owner[last] == self._turn:
                captured = succ._seeds[succ._board.opposite[last]]
                succ._seeds[succ._board.stores[self._turn]] += (1 + captured)
                succ._seeds_left[self._turn] -= 1
                succ._seeds_left[1 - self._turn] -= captured
                succ._seeds[last] = 0
                succ._seeds[succ._board.opposite[last]] = 0

            # free turn if ending in store
            if last == succ._board.stores[self._turn]:
                succ._turn = self._turn
            else:
                succ._turn = 1 - self._turn

            # game is over if one player has no seeds left
            if succ._seeds_left[0] == 0 or succ._seeds_left[1] == 0:
                for p in range(0, 2):
                    succ._seeds[succ._board.stores[p]] += succ._seeds_left[p]
                    succ._seeds_left[p] = 0
                for p in range(0, succ._board.size):
                    if succ._board.owner[p] is not None:
                        succ._seeds[p] = 0

            succ._compute_hash()
                
            return succ

        
        def is_terminal(self):
            ''' Determines if this state is terminal -- whether the game is over having
                reached this state.
            '''
            return sum(self._seeds_left) == 0

        
        def _winner(self):
            return self.winner()

        
        def payoff(self):
            ''' Returns the payoff to player 0 at this state: 1 for a win, 0 for a draw, -1 for
                a loss.  The return value is None if this state is not terminal.
            '''
            if not self.is_terminal():
                return None
            else:
                difference = self._seeds[self._board.stores[0]] - self._seeds[self._board.stores[1]]
                return (difference > 0) - (difference < 0)

            
        def _seeds_stored(self, p):
            ''' Returns the number of seeds in the store for the given player.
               
                p -- the index of a player; either 0 or 1
            '''
            return self._seeds[self._board.stores[p]]

        
        def __str__(self):
            result = []
            if self._turn == 1:
                result.append('> ')
            else:
                result.append('  ')
            for i in range(0, self._board.pits + 1):
                result.append('%2d ' % self._seeds[self._board.size - 1 - i])
            result.append('   ')
            result.append('\n')
            if self._turn == 0:
                result.append('>    ')
            else:
                result.append('     ')
            for i in range(0, self._board.pits + 1):
                result.append('%2d ' % self._seeds[i])
            result.append('\n')
            result.append('%d %d' % (self._seeds_left[0], self._seeds_left[1]))
            return "".join(result)

        
        def __repr__(self):
            return '%r %r %r' % (self._seeds, self._seeds_left, self._turn)

        
        def _compute_hash(self):
            # faster hash computation; thanks to CF
            self.hash = hash(tuple(self._seeds)) * 2 + self._turn

            
        def __hash__(self):
            return self.hash

        
        def __eq__(self, other):
            return isinstance(other, self.__class__) and self._seeds == other._seeds and self._turn == other._turn and self._board is other._board


if __name__ == '__main__':
    board = Kalah(6)
    pos = board.initial_state(4)
    print(pos)
    for i in pos.get_actions():
        print("move again: ", pos.is_move_again(i))
    for i in pos.get_actions():
        print("capture: ", pos.is_capture(i))
    pos = pos.successor(4)
    pos = pos.successor(7)
    print(pos)
    for i in pos.get_actions():
        print("move again: ", pos.is_move_again(i))
    for i in pos.get_actions():
        print("capture: ", pos.is_capture(i))

    
