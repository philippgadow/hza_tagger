"""Unit tests for truth matching using synthetic awkward arrays.

No ROOT files needed — we fabricate NanoAOD-shaped events.
"""

import numpy as np
import awkward as ak
import pytest

from common.truth_matching import label_jets, find_a_bosons


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_genparts(
    etas, phis, pdgids, mothers, status_flags
):
    return (
        ak.Array(etas),
        ak.Array(phis),
        ak.Array(pdgids),
        ak.Array(mothers),
        ak.Array(status_flags),
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestFindABosons:
    def test_finds_last_copy(self):
        # bit 13 set = isLastCopy (1<<13 = 8192)
        pdg   = ak.Array([[36, 36, 21]])
        flags = ak.Array([[8192, 0, 8192]])  # first is last copy, second isn't
        mask  = find_a_bosons(pdg, flags)
        assert ak.to_list(mask) == [[True, False, False]]

    def test_no_a_boson(self):
        pdg   = ak.Array([[21, 1, -1]])
        flags = ak.Array([[8192, 8192, 8192]])
        mask  = find_a_bosons(pdg, flags)
        assert not ak.any(mask)


class TestLabelJets:
    def _single_event(self, jet_etas, jet_phis, gen_etas, gen_phis,
                      gen_pdgids, gen_mothers, gen_flags):
        result = label_jets(
            jet_eta=ak.Array([jet_etas]),
            jet_phi=ak.Array([jet_phis]),
            gen_eta=ak.Array([gen_etas]),
            gen_phi=ak.Array([gen_phis]),
            gen_pdgid=ak.Array([gen_pdgids]),
            gen_mother_idx=ak.Array([gen_mothers]),
            gen_status_flags=ak.Array([gen_flags]),
        )
        return result["labels"]

    def test_jet_near_a_is_labeled_signal(self):
        # a at (eta=0, phi=0) with daughters at (0.1, 0) and (0.2, 0)
        # jet at (0.05, 0) should match
        labels = self._single_event(
            jet_etas=[0.05],   jet_phis=[0.0],
            gen_etas =[0.0,    0.1,  0.2],
            gen_phis =[0.0,    0.0,  0.0],
            gen_pdgids=[36,     2,    -2],       # a, quark, antiquark
            gen_mothers=[-1,    0,    0],         # daughters of idx 0 (the a)
            gen_flags=[8192, 8192, 8192],
        )
        assert ak.to_list(labels) == [[1]]

    def test_jet_far_from_a_is_background(self):
        labels = self._single_event(
            jet_etas=[2.5],    jet_phis=[0.0],
            gen_etas =[0.0,    0.1,  0.2],
            gen_phis =[0.0,    0.0,  0.0],
            gen_pdgids=[36,     2,    -2],
            gen_mothers=[-1,    0,    0],
            gen_flags=[8192, 8192, 8192],
        )
        assert ak.to_list(labels) == [[0]]

    def test_multiple_jets_only_one_signal(self):
        labels = self._single_event(
            jet_etas=[0.05, 2.5],  jet_phis=[0.0, 0.0],
            gen_etas =[0.0,    0.1,  0.2],
            gen_phis =[0.0,    0.0,  0.0],
            gen_pdgids=[36,     2,    -2],
            gen_mothers=[-1,    0,    0],
            gen_flags=[8192, 8192, 8192],
        )
        result = ak.to_list(labels)[0]
        assert result == [1, 0]

    def test_no_a_boson_all_background(self):
        labels = self._single_event(
            jet_etas=[0.0, 1.0],  jet_phis=[0.0, 0.0],
            gen_etas =[0.0],
            gen_phis =[0.0],
            gen_pdgids=[21],      # just a gluon
            gen_mothers=[-1],
            gen_flags=[8192],
        )
        result = ak.to_list(labels)[0]
        assert result == [0, 0]

    def test_daughter_outside_cone_not_matched(self):
        # a at (0, 0) but one daughter is far away (dR > 0.4)
        labels = self._single_event(
            jet_etas=[0.05],   jet_phis=[0.0],
            gen_etas =[0.0,    0.1,  2.5],   # second daughter far
            gen_phis =[0.0,    0.0,  0.0],
            gen_pdgids=[36,     2,    -2],
            gen_mothers=[-1,    0,    0],
            gen_flags=[8192, 8192, 8192],
        )
        assert ak.to_list(labels) == [[0]]

    # ── multi-boson regression tests ─────────────────────────────────────────

    def test_two_bosons_condition_not_cross_satisfied(self):
        # Regression: jet near boson A (idx 0), but A's second daughter is outside
        # the cone (eta=2.5). Boson B (idx 3) is far away; its daughters happen to
        # sit near the jet. Old code labelled this as signal (false positive).
        labels = self._single_event(
            jet_etas=[0.05],  jet_phis=[0.0],
            #           boson A          dau A1 (in)   dau A2 (OUT)
            gen_etas =[0.0,              0.1,           2.5,
            #           boson B (far)    dau B1 (in)   dau B2 (in)
                       3.0,              0.15,          0.25],
            gen_phis =[0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            gen_pdgids=[36,   2,   -2,   36,    2,    -2],
            gen_mothers=[-1,  0,    0,   -1,    3,     3],
            gen_flags=[8192]*6,
        )
        assert ak.to_list(labels) == [[0]]

    def test_two_bosons_correct_match_first_boson(self):
        # Jet matches boson A (both daughters inside cone). Boson B is far away.
        labels = self._single_event(
            jet_etas=[0.05],  jet_phis=[0.0],
            gen_etas =[0.0,   0.1,  0.2,   3.0,  3.1,  3.2],
            gen_phis =[0.0,   0.0,  0.0,   0.0,  0.0,  0.0],
            gen_pdgids=[36,    2,   -2,    36,    2,    -2],
            gen_mothers=[-1,   0,    0,    -1,    3,     3],
            gen_flags=[8192]*6,
        )
        assert ak.to_list(labels) == [[1]]

    def test_two_bosons_truth_mass_from_matched_boson(self):
        # truth_a_mass must be the mass of the boson that triggered label=1 (A),
        # not boson B which is far away.
        mass_A, mass_B = 1.0, 2.0
        result = label_jets(
            jet_eta=ak.Array([[0.05]]),
            jet_phi=ak.Array([[0.0]]),
            gen_eta=ak.Array([[0.0, 0.1, 0.2, 3.0, 3.1, 3.2]]),
            gen_phi=ak.Array([[0.0]*6]),
            gen_pdgid=ak.Array([[36, 2, -2, 36, 2, -2]]),
            gen_mother_idx=ak.Array([[-1, 0, 0, -1, 3, 3]]),
            gen_status_flags=ak.Array([[8192]*6]),
            gen_mass=ak.Array([[mass_A, 0.0, 0.0, mass_B, 0.0, 0.0]]),
        )
        assert ak.to_list(result["labels"])[0] == [1]
        assert abs(ak.to_list(result["truth_a_mass"])[0][0] - mass_A) < 1e-5

    def test_two_bosons_both_match_closest_mass_wins(self):
        # Both bosons satisfy the matching conditions. truth_a_mass must come
        # from the boson whose axis is closer to the jet.
        mass_close, mass_far = 0.5, 2.5
        result = label_jets(
            jet_eta=ak.Array([[0.05]]),
            jet_phi=ak.Array([[0.0]]),
            # boson A axis at eta=0.0 (dR≈0.05), boson B axis at eta=0.2 (dR≈0.15)
            gen_eta=ak.Array([[0.0, 0.1, 0.15,   0.2, 0.3, 0.1]]),
            gen_phi=ak.Array([[0.0]*6]),
            gen_pdgid=ak.Array([[36, 2, -2, 36, 2, -2]]),
            gen_mother_idx=ak.Array([[-1, 0, 0, -1, 3, 3]]),
            gen_status_flags=ak.Array([[8192]*6]),
            gen_mass=ak.Array([[mass_close, 0.0, 0.0, mass_far, 0.0, 0.0]]),
        )
        assert ak.to_list(result["labels"])[0] == [1]
        assert abs(ak.to_list(result["truth_a_mass"])[0][0] - mass_close) < 1e-5

    def test_a_boson_leptonic_decay_skipped(self):
        # a decays only to e+e- (leptonic). A jet near the a must not be labelled.
        labels = self._single_event(
            jet_etas=[0.05],  jet_phis=[0.0],
            gen_etas =[0.0,   0.1,  -0.1],
            gen_phis =[0.0,   0.0,   0.0],
            gen_pdgids=[36,   11,   -11],   # a → e+ e-
            gen_mothers=[-1,   0,    0],
            gen_flags=[8192]*3,
        )
        assert ak.to_list(labels) == [[0]]
