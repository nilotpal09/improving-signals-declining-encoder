"""Uncentred kappa for the epoch-100 response figure (2026-10-10).

Show each metric family as the survey computed it. For kappa that is the uncentred
condition number (the embedding matrix as given);
knee_response_e100.py recorded our centred variant. Same runs, rules, offsets and output format as
knee_response_e100.py (its main() is reused unchanged).
Writes results/knee_response_e100_kappa.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_response_e100 as E  # noqa: E402

E.OUT = os.path.join(paths.CSV, 'knee_response_e100_kappa.csv')
E.SERIES = ('readability R2 average', 'kappa uncentred')

if __name__ == '__main__':
    E.main()
