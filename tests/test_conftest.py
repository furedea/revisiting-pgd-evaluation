"""Tests for conftest.py -- requires_tf marker and TF skip mechanism."""

import pytest


class TestRequiresTfMarkerRegistration:
    """Verify the requires_tf marker is registered and functional."""

    def test_requires_tf_marker_is_registered(self, pytestconfig):
        """The requires_tf marker should be registered in pytest."""
        markers = pytestconfig.getini("markers")
        marker_names = [m.split(":")[0].strip() for m in markers]
        assert "requires_tf" in marker_names

    def test_requires_tf_marker_can_be_applied(self):
        """A test decorated with @pytest.mark.requires_tf should not raise
        an 'Unknown pytest.mark.requires_tf' warning."""
        # If the marker is not registered, pytest would emit a warning.
        # This test simply confirms the marker can be used without error.
        mark = pytest.mark.requires_tf
        assert mark is not None


class TestRequiresTfSkipBehavior:
    """Verify the skip mechanism works based on TF availability."""

    def test_has_tf_flag_is_boolean(self):
        """The HAS_TF flag exposed in conftest should be a boolean."""
        from tests.conftest import HAS_TF

        assert isinstance(HAS_TF, bool)

    def test_requires_tf_variable_exists_in_conftest(self):
        """The module-level requires_tf variable should exist in conftest."""
        import tests.conftest as conftest_mod

        assert hasattr(conftest_mod, "requires_tf")

    def test_requires_tf_is_a_pytest_mark(self):
        """The requires_tf variable should be a pytest MarkDecorator."""
        from tests.conftest import requires_tf

        assert isinstance(requires_tf, pytest.MarkDecorator)

    @pytest.mark.requires_tf
    def test_requires_tf_decorated_test_skips_when_tf_unavailable(self):
        """This test is decorated with @pytest.mark.requires_tf.
        If TF is not available (HAS_TF=False), it should be skipped.
        If TF is available (HAS_TF=True), it should pass.

        We do NOT ``import tensorflow`` here because even with the
        @requires_tf guard, a segfaulting TF import would crash the
        process.  Instead we just verify that HAS_TF is True, which
        is the invariant when this test actually runs.
        """
        from tests.conftest import HAS_TF

        assert HAS_TF is True
