"""NE-Sum and DSE for the epoch-100 response figure (2026-10-10).

All seven metric families of the survey must be visible in the figure; knee_response_e100.py covered five
(alpha-ReQ, RankMe, kappa, Self-Cluster, TwoNN). Same runs, rules, offsets and output format as
knee_response_e100.py (its main() is reused unchanged), for every recorded variant of the other two:
NE-Sum centred / uncentred; DSE code, paper, paper_codebw, code_l2, code_l2_centred (definitions in the
paper's Table tab:metricconventions; preference: higher for both).
Writes results/knee_response_e100_ext.csv (refuses to overwrite).
"""
import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
sys.path.insert(0, _HERE)
import knee_response_e100 as E  # noqa: E402

E.OUT = os.path.join(paths.CSV, 'knee_response_e100_ext.csv')
E.SERIES = ('readability R2 average', 'ne_sum centred', 'ne_sum uncentred', 'dse code', 'dse paper',
            'dse paper_codebw', 'dse code_l2', 'dse code_l2_centred')

if __name__ == '__main__':
    E.main()
