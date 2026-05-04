import numpy as np
import pytest

from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.data.simulation import generate_multiparticle_kymograph
from kymo_tracker.deeplearning.predict import (
    create_mask_from_centers_widths,
    extract_trajectories_from_heatmap,
)
from kymo_tracker.utils.helpers import generate_kymograph


def test_multitask_dataset_rejects_single_track_configuration():
    with pytest.raises(ValueError, match="min_trajectories"):
        MultiTaskDataset(n_samples=1, min_trajectories=1)


def test_multitask_dataset_emits_standard_multitrack_tensors():
    dataset = MultiTaskDataset(
        n_samples=1,
        min_trajectories=2,
        max_trajectories=3,
        seed=123,
    )

    noisy, true_noise, target = dataset[0]

    assert noisy.shape == (1, 16, 512)
    assert true_noise.shape == (1, 16, 512)
    assert target.shape == (1, 16, 512)
    assert np.any(target.squeeze(0).numpy() > 0.5)


def test_simulation_rejects_single_particle_inputs():
    with pytest.raises(ValueError, match="at least two particles"):
        generate_multiparticle_kymograph(
            contrast=[0.8],
            diffusion=[1.0],
        )

    with pytest.raises(ValueError, match="at least two particles"):
        generate_kymograph(
            contrast=np.float32(0.8),
            diffusion=np.float32(1.0),
        )


def test_simulation_returns_multiparticle_paths():
    noisy, gt, paths = generate_multiparticle_kymograph(
        length=8,
        width=32,
        contrast=[0.8, 0.6],
        diffusion=[1.0, 0.8],
        seed=123,
    )

    assert noisy.shape == (8, 32)
    assert gt.shape == (8, 32)
    assert paths.shape == (2, 8)


def test_heatmap_extraction_preserves_configured_track_count_when_empty():
    kymograph = np.ones((4, 32), dtype=np.float32)
    heatmap = np.zeros((4, 32), dtype=np.float32)

    trajectories = extract_trajectories_from_heatmap(
        kymograph,
        heatmap,
        max_tracks=3,
    )

    assert len(trajectories) == 3
    assert all(traj.shape == (4,) for traj in trajectories)
    assert all(np.isnan(traj).all() for traj in trajectories)


def test_center_width_mask_requires_multiple_tracks():
    centers = np.array([5.0, 6.0], dtype=np.float32)
    widths = np.array([3.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        create_mask_from_centers_widths(centers, widths, (2, 32))

    single_track_centers = centers[:, np.newaxis]
    single_track_widths = widths[:, np.newaxis]
    with pytest.raises(ValueError, match="at least two tracks"):
        create_mask_from_centers_widths(
            single_track_centers,
            single_track_widths,
            (2, 32),
        )


def test_center_width_mask_accepts_multiple_tracks():
    centers = np.array([[5.0, 15.0], [6.0, 16.0]], dtype=np.float32)
    widths = np.full_like(centers, 3.0)

    mask, labels = create_mask_from_centers_widths(centers, widths, (2, 32))

    assert mask.shape == (2, 32)
    assert labels.shape == (2, 32)
    assert labels.max() == 2
