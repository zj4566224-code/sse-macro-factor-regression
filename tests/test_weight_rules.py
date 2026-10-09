from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_analysis import PHASES, add_weights, parse_args  # noqa: E402


class WeightRuleTest(unittest.TestCase):
    def test_confirmed_boundaries_and_phase_weights(self):
        with patch.object(sys, "argv", ["run_analysis.py"]):
            args = parse_args()
        frame = pd.DataFrame(
            {
                "phase": [PHASES[0][0], PHASES[1][0], PHASES[2][0], PHASES[0][0]],
                "spread": [0.50, 0.75, 0.49, 0.76],
            }
        )
        weighted = add_weights(frame, args)

        self.assertEqual(weighted["spread_weight"].tolist(), [0.5, 0.5, 1.7, 1.7])
        self.assertEqual(weighted["phase_weight"].tolist(), [1.65, 1.55, 1.65, 1.65])
        self.assertAlmostEqual(weighted.loc[0, "weight"], 1.65 * 0.5)
        self.assertAlmostEqual(weighted.loc[1, "weight"], 1.55 * 0.5)
        self.assertAlmostEqual(weighted.loc[2, "weight"], 1.65 * 1.7)


if __name__ == "__main__":
    unittest.main()
