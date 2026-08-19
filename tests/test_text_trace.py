import tempfile
import unittest

import numpy as np

from PythIon.IO import _load_text_trace


class TextTraceTest(unittest.TestCase):
    def test_loads_one_pa_value_per_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as trace:
            trace.write("4263.30022\n4270.00089\n")
            trace.flush()
            np.testing.assert_allclose(
                _load_text_trace(trace.name), [4.26330022e-9, 4.27000089e-9]
            )


if __name__ == "__main__":
    unittest.main()
