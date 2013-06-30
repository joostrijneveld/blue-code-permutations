#! /usr/bin/env python

from itertools import combinations, izip_longest, product, chain, count
from copy import deepcopy
from collections import Counter
import glob
import time

def powerset(s):
    """ Itertools recipe. Returns the powerset of a list. """
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def grouper(iterable, n, fillvalue=None):
    """Itertools recipe. Collect data into fixed-length chunks or blocks"""
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class TwoWayDict(dict):

    def __init__(self, iterable):
        dict.__init__(self)
        for (c, n) in zip(iterable, count(1)):
            self.__setitem__(c, n)
            self.__setitem__(n, c)

    def __len__(self):
        return dict.__len__(self) / 2

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

MEMORYPATH = "plaintext/*"  # telegrams that provide code groups for frequency count
TELEGRAMPATHS = ("telegrams/2 301001071", "telegrams/3 304001200")  # Finds July 1st permutation
# TELEGRAMPATHS = ("telegrams/8 409002114", "telegrams/4 307202231b") # Finds 02-09
# TELEGRAMPATHS = ("telegrams/13 310003155", "telegrams/14 297003153") # Finds 03-09
# TELEGRAMPATHS = ("telegrams/17 029005170", "telegrams/20 320005191") # Finds 05-09

ALPHABET = TwoWayDict(('A', 'E', 'Ha', 'He', 'Hi', 'Ho', 'Fu', 'I', 'Ka', 'Ke', 'Ki',
            'Ko', 'Ku', 'Ma', 'Me', 'Mi', 'Mo', 'Mu', 'Na', 'Ne', 'Ni', 'No',
            'O', 'Ra', 'Re', 'Ri', 'Ro', 'Ru', 'Sa', 'Se', 'Shi', 'So', 'Su',
            'Ta', 'Te', 'Chi', 'To', 'Tsu', 'U', 'Wa', 'Wi', 'Ya', 'Yo', 'Yu'))
ALPHACOUNT = 44

NUBOERGROUPS = (('Sa', 'He', 'Sa', 'Mo'),
                ('Ka', 'Ni', 'Ta', 'Na'),
                ('Ki', 'Ke', 'A', 'Ni'),
                ('Mi', 'I', 'Wa', 'Na'),
                ('Fu', 'Me', 'Ku', 'Ta'),
                ('Mo', 'Na', 'Ta', 'Re'),
                ('Mi', 'Ru', 'No', 'Ni'),
                ('Mo', 'Ko', 'Ru', 'Ko'),
                ('A', 'Shi', 'A', 'So'),
                ('Mi', 'I', 'Ka', 'So'),
                ('Ha', 'Wi', 'Fu', 'Ho'),
                ('Ro', 'Ne', 'Ha', 'Hi'),
                ('Ma', 'Sa', 'No', 'Ne'),
                ('Ra', 'So', 'Ra', 'Te'))

fT =   [['n',' ','x',' ','n',' ','x',' ',' ',' ',' ',],
        [' ','n',' ','n',' ','n',' ','n',' ','x',' ',],
        [' ',' ',' ',' ',' ',' ','x',' ',' ','x','x',],
        ['x',' ','x',' ',' ',' ',' ','x',' ',' ',' ',],
        [' ',' ','x',' ',' ',' ','x',' ',' ',' ',' ',],
        [' ','x',' ',' ',' ','x',' ',' ',' ',' ',' ',],
        [' ',' ',' ',' ',' ',' ','x',' ',' ',' ','x',],
        [' ',' ',' ','x','x','x',' ',' ',' ','x',' ',],
        ['x',' ',' ',' ',' ',' ',' ',' ',' ','x',' ',],
        ['x',' ',' ',' ',' ',' ',' ','x',' ','x','x',],
        [' ','x',' ',' ',' ',' ','x','x','x','x','x',]]

f = [list(col) for col in zip(*fT)]  # much easier to work with in columns as opposed to rows
memory = Counter()

# computing several constants based on the figure
COLNUM = len(f)
ROWNUM = len(fT)
COLLEN = [ROWNUM - col.count('x') for col in f]
FIGLEN = sum(col.count(' ') for col in f)
SPACES = FIGLEN + sum(col.count('n') for col in f)
GCOUNT = FIGLEN / 4

def precompute_colcombs():
    d = dict()
    for cols in powerset(zip(range(COLNUM), COLLEN)):
        colsum = sum([b for _, b in cols])
        if colsum not in d:
            d[colsum] = []
        d[colsum].append(set([a for a, _ in cols]))
    return d

COLCOMBS = precompute_colcombs()

def kana2int(c):
    return 0 if c == "?" or '.' in c else ALPHABET[c]
    
def int2kana(i):
    return '..' if i == 0 else ALPHABET[i]

def symbol_space(col):
    return col.count(' ') + col.count('n')

def checksum(a, b, c):
    x = (a + b + c - 1) % 44 
    return x if x > 0 else 44  # since 0 isn't valid, but 44 is

def missing_symbol(a, b, x):
    """ Computes one of the three code group symbols based on the other two (a, b) and checksum x """
    c = x - a - b + 1
    while c <= 0:
        c += 44
    return c

def find_groups():
    i = 0
    groups = []
    for g in xrange(1, GCOUNT + 1):
        group = []
        while len(group) < 4:
            x, y = i % COLNUM , i / COLNUM
            if f[x][y] == 'x' or f[x][y] == 'n':
                i += 1
            else:
                group.append((x, y))
                i += 1
        groups.append((g, group))
    return groups

##### Functions for recognising common groups

def remember_groups(f):
    for l in f.readlines():
        for t in grouper(l.split(), 4):
            if ".." not in t:
                a, b, c, x = map(kana2int, t)
                if checksum(a, b, c) is x:
                    memory.update([(a, b, c, x)])  # list of tuples, as iterable is required

def parse_telegrams():
    telegrams = []
    tfiles = [open(fname, 'r') for fname in TELEGRAMPATHS]
    for tgram in tfiles:
        content = tgram.readline().split()
        if len(content) < SPACES:
            raise Exception("Supplied telegram is too short")
        telegrams.append(map(kana2int, content[:SPACES]))
    return telegrams

def find_positions(telegram):
    positions = [[] for i in range(ALPHACOUNT)]
    for (n, i) in zip(telegram, xrange(SPACES)):
        positions[n % 44].append(i)
    return positions

def identify_group(telegram, positions):
    groups = [a for a, _ in memory.most_common()]
    for (g, x) in memory.most_common():
        p = [positions[n%44] for n in g]
        for pos in product(*p):  # all combinations of positions
            if checksum(*[telegram[i] for i in pos[:3]]) == telegram[pos[3]]:
                if tuple([telegram[i] for i in pos[:4]]) in groups:
                    yield pos

##### End recognising common groups
##### Functions for finding valid groups

def valid_colset_pair((cset1, cset2), pairedcols):
    """ Tests if two pairs of column-sets are simultaniously possible
    cset1, cset2 -- tuples of form (c1, [ci], cn) where c1 and cn are edges
    pairedcols   -- iterable containing pairs of neighbouring columns """
    c1a, cia, cna = cset1
    c1b, cib, cnb = cset2
    seta = set([c1a, cna]) | cia
    setb = set([c1b, cnb]) | cib
    if not seta.isdisjoint(setb):
        for a, b in pairedcols:
            if cna == a and c1b == b:
                return False
            if cnb == a and c1a == b:
                return False
        if c1a in setb and cna in setb and not seta <= setb:
            return False
        if c1b in seta and cnb in seta and not setb <= seta:
            return False
        if cna not in set([cnb]) | cib and cnb not in set([cna]) | cia:
            return False
        if c1a not in set([c1b]) | cib and c1b not in set([c1a]) | cia:
            return False
    return True

def valid_colset_comb(prod, pairedcols):
    """ Checks if a series of column-sets are simultaniously possible
    prod       -- iterable of form [(c1, [ci], cn)] where c1 and cn are edges
    pairedcols -- iterable containing pairs of neighbouring columns """
    for c1, ci, cn in prod:
        cols = set([c1, cn]) | ci
        for a, b in pairedcols:
            if (a in ci and b not in ci and b != cn) or (b in ci and a not in ci and a != c1):
                return False
            if (a == c1 and b not in ci) or (a == cn and b in cols):
                return False
            if (b == c1 and a in cols) or (b == cn and a not in ci):
                return False
    return all(valid_colset_pair(x, pairedcols) for x in combinations(prod, 2))

def valid_group(group, pos):
    """Checks if the supplied group conforms to distances of pos, given the figure
    group -- tuple of four (x,y) coordinates
    pos   -- tuple of four positions within the telegram"""
    pairs = []
    for (s1, s2) in combinations((0, 1, 2, 3), 2):
        if pos[s2] < pos[s1]:
            s1, s2 = s2, s1
        pairs.append((pos[s2] - pos[s1], s1, s2))
    hardpairs = []
    pairedcols = []
    for (d, s1, s2) in pairs:
        col1 = f[group[s1][0]]
        col2 = f[group[s2][0]]
        if group[s1][0] == group[s2][0]:  # if it concerns symbols in the same column
            dist = symbol_space(col1[group[s1][1]:group[s2][1]])
            if dist != d:
                return False
            continue
        dist = symbol_space(col1[group[s1][1]:]) + symbol_space(col2[:group[s2][1]])
        if dist > d:
            return False  # if the actual distance is larger than required, it is impossible
        elif dist < d:
            if d - dist not in COLCOMBS:
                return False
            hardpairs.append((d - dist, group[s1][0], group[s2][0]))  # put off the difficult ones for later
        else:
            pairedcols.append((group[s1][0], group[s2][0]))  # perfect fit, need to be neighbouring sequences
    if not hardpairs:
        return True
    usedcols = []
    for (d, c1, c2) in hardpairs:
        colsets = [(c1, x, c2) for x in COLCOMBS[d] if c1 not in x and c2 not in x]
        if not colsets:
            return False
        usedcols.append(colsets)
    return any(valid_colset_comb(prod, pairedcols) for prod in product(*usedcols))

##### End finding valid groups
##### Functions for deduction

def fill_colpart(fig, x, y, v, d, lim, used):
    c = 0
    for j in xrange(lim):
        if fig[x][y + j * d] != 'x':
            fig[x][y + j * d] = v + c * d
            used[v + c * d] = True
            c += 1

def fill_col(fig, x, y, v, used, usedcols):
    usedcols[x] = True
    fill_colpart(fig, x, y, v, 1, ROWNUM - y, used)
    fill_colpart(fig, x, y, v, -1, y + 1, used)

def valid_symbol(n, na, nb, used, usedcols):
    """ Checks to see if the symbol at position n in the telegram is a valid candidate w.r.t. the column it is in
    n        -- the position in the telegram
    na       -- the required space before the symbol
    nb       -- the required space after the symbol
    used     -- list denoting used/unused, per symbols in the telegram
    usedcols -- list denoting which columns have been filled"""
    if n - na < 0 or n + nb >= SPACES:
        return False
    if any([used[x] for x in range(n - na, n + nb + 1)]):
        return False
    ca, cb = 0, 0
    for x in range(n - na - 1, -1, -1):
        if used[x]:
            break
        ca += 1
    for x in range(n + nb + 1, SPACES):
        if used[x]:
            break
        cb += 1
    if (ca > 0 and ca not in COLCOMBS) or (cb > 0 and cb not in COLCOMBS):
        return False
    if ca > 0 and not [x for x in COLCOMBS[ca] if not any(usedcols[c] for c in x)]:
        return False
    if cb > 0 and not [x for x in COLCOMBS[cb] if not any(usedcols[c] for c in x)]:
        return False
    return True

def print_fig(fig):
    fix_len = lambda x: str(x) + ' ' * (2 - len(str(x))) if isinstance(x, int) else x + ' '
    for a in [[fix_len(x) for x in row] for row in zip(*fig)]:
        print a

def find_permutation(fig):
    """ Extracts the permutation from a filled figure"""
    permutation = zip([next(x for x in xs if isinstance(x, int)) for xs in fig], count(1))
    permutation.sort()
    return [x for _, x in permutation]

def deduce(fig, pos, validgroups, groups, telegram):
    """ Tries to fill the figure by deduction
    fig         -- an unfilled copy of the figure
    pos         -- the position in the telegram of the supposed 'common' group
    validgroups -- positions where it can be positioned
    groups      -- a list of positions of groups in the figure
    telegram    -- the reference telegram, as a list of symbols """
    for _, l in validgroups:
        used = [False] * SPACES
        usedcols = [False] * COLNUM
        for i in range(4):
            fig[l[i][0]][l[i][1]] = pos[i]
            fill_col(fig, l[i][0], l[i][1], pos[i], used, usedcols)
            for x in filter(lambda x: isinstance(x, int), fig[l[i][0]]):
                used[x] = True
        todo_flag = True
        while todo_flag:
            todo_flag = False
            queue = []
            for _, group in groups:
                ps = filter(lambda (x, y): isinstance(fig[x][y], int), group)
                if len(ps) == 3:
                    missing = [i for (x, y), i in zip(group, count()) if not isinstance(fig[x][y], int)][0]
                    if missing == 3:
                        z = checksum(*[telegram[fig[x][y]] for (x, y) in ps])
                    else:
                        z = missing_symbol(*[telegram[fig[x][y]] for (x, y) in ps])
                    queue.append((group[missing], z))
            while queue:
                for ((x, y), z) in queue:
                    candidates = [n for n in range(SPACES) if telegram[n] == z]
                    na = symbol_space(f[x][:y])
                    nb = symbol_space(f[x][y + 1:])
                    candidates = filter(lambda n: valid_symbol(n, na, nb, used, usedcols), candidates)
                    if len(candidates) == 1:
                        fig[x][y] = candidates[0]
                        fill_col(fig, x, y, candidates[0], used, usedcols)
                        todo_flag = True
                        break
                else:
                    break
        print_fig(fig)
        if any([' ' in col for col in fig]):
            print "Figure incomplete"
        else:
            print "Permutation found!", find_permutation(fig)
            break

##### End deduction

def main():
    t0 = time.clock()
    for fname in glob.glob(MEMORYPATH):
        remember_groups(open(fname, 'r'))
    d = dict((tuple(map(kana2int, x)), 50) for x in NUBOERGROUPS)
    memory.update(d)
    telegrams = parse_telegrams()
    for t in range(2):
        print "Using telegram " + str(t + 1) + " as a base.."
        positions = find_positions(telegrams[t])
        group_gen = identify_group(telegrams[1 - t], positions)
        groups = find_groups()
        for pos in group_gen:
            validgroups = [(g, group) for (g, group) in groups if valid_group(group, pos)]
            print "  Trying", pos
            if validgroups:
                fig = deepcopy(f)
                deduce(fig, pos, validgroups, groups, telegrams[t])
    print time.clock() - t0, "seconds process time"

if __name__ == "__main__":
    main()
    