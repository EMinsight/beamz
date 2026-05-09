import pytest

from tests.test_mode_source import (
    TestModeSourceDirectionality3D,
    TestModeSourceEffectiveIndex,
    TestModeSourcePolarization,
    TestModeSourceProfile,
    TestModeSourcePropagation,
)

pytestmark = [pytest.mark.integration, pytest.mark.simulation]

TestModeSourceEffectiveIndex.__test__ = True
TestModeSourceProfile.__test__ = True
TestModeSourcePropagation.__test__ = True
TestModeSourcePolarization.__test__ = True
TestModeSourceDirectionality3D.__test__ = True
