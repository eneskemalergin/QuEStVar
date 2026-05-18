from __future__ import annotations

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
	try:
		yield
	finally:
		plt.close("all")
