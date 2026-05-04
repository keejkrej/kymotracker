"""Main demo script that runs training and inference, then creates comparison plots."""

import numpy as np
from pathlib import Path
import sys

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.utils.helpers import (
    get_diffusion_coefficient,
)
from kymo_tracker.data.simulation import generate_multiparticle_kymograph
from kymo_tracker.classical.pipeline import classical_median_threshold_tracking
from kymo_tracker.deeplearning.training.config import (
    DEFAULT_EPOCHS,
    DEFAULT_MASK_PEAK_WIDTH_SAMPLES,
    DEFAULT_WINDOW_LENGTH,
    DEFAULT_WIDTH,
    DEFAULT_RADII_NM,
    DEFAULT_CONTRAST,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_MIN_TRAJECTORIES,
    DEFAULT_MAX_TRAJECTORIES,
)
from kymo_tracker.deeplearning.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
    load_multitask_model,
)
from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.deeplearning.predict import (
    process_slice_independently,
    link_trajectories_across_slices,
)
from kymo_tracker.utils.device import get_default_device
from kymo_tracker.utils.visualization import visualize_comparison


def run_classical_pipeline(kymograph_noisy):
    """Run classical median filter + thresholding pipeline."""
    result = classical_median_threshold_tracking(
        kymograph_noisy,
        median_kernel=(11, 11),
        threshold_mode="otsu",
        min_component_size=50,  # Increased from 8 to filter out small noise regions
    )
    
    trajectories = list(result.trajectories)
    
    # Create combined mask from all instance masks
    combined_mask = np.zeros_like(kymograph_noisy, dtype=bool)
    for instance_mask in result.instance_masks:
        combined_mask |= instance_mask
    
    return {
        'filtered': result.filtered,
        'mask': combined_mask,
        'labeled_mask': result.labeled_mask,
        'trajectories': trajectories,
    }


def run_deeplearning_pipeline(kymograph_noisy, model, device):
    """Run deep learning denoising + locator pipeline."""
    T, W = kymograph_noisy.shape
    chunk_size = 16
    overlap = 8
    
    # Process each slice independently
    slice_results = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        slice_data = kymograph_noisy[start:end]
        slice_result = process_slice_independently(model, slice_data, device=device)
        slice_results.append(slice_result)
        start += chunk_size - overlap
    
    # Link trajectories
    slice_trajectories_list = [result['trajectories'] for result in slice_results]
    linked_trajectories = link_trajectories_across_slices(
        slice_trajectories_list,
        chunk_size=chunk_size,
        overlap=overlap,
        total_length=T,
    )
    
    # Reconstruct full outputs for visualization
    denoised_full = np.zeros((T, W), dtype=np.float32)
    heatmap_full = np.zeros((T, W), dtype=np.float32)
    weights = np.zeros((T, W), dtype=np.float32)
    
    window = np.ones(chunk_size)
    if overlap > 0:
        fade_len = overlap // 2
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    start = 0
    for result in slice_results:
        end = min(start + chunk_size, T)
        actual_len = end - start
        
        weight_chunk = window[:actual_len, np.newaxis]
        denoised_full[start:end] += result['denoised'] * weight_chunk
        heatmap_full[start:end] += result['heatmap'] * weight_chunk
        weights[start:end] += weight_chunk
        
        start += chunk_size - overlap
    
    denoised_full = np.divide(denoised_full, weights, out=np.zeros_like(denoised_full), where=weights > 0)
    heatmap_full = np.divide(heatmap_full, weights, out=np.zeros_like(heatmap_full), where=weights > 0)
    n_tracks = len(linked_trajectories)
    centers_full = np.full((T, n_tracks), np.nan, dtype=np.float32)
    widths_full = np.full((T, n_tracks), np.nan, dtype=np.float32)

    for track_idx, traj in enumerate(linked_trajectories):
        traj_array = np.asarray(traj)
        actual_len = min(T, len(traj_array))
        centers_full[:actual_len, track_idx] = traj_array[:actual_len]
    
    return {
        'denoised': denoised_full,
        'heatmap': heatmap_full,
        'trajectories': linked_trajectories,
        'centers': centers_full,
        'widths': widths_full,
    }


def generate_demo_cases():
    """Generate 5 test cases with different scenarios."""
    cases = []
    
    # Case 1: Two particles, low noise
    print("Generating case 1: Two particles, low noise...")
    radii = [10.0, 18.0]
    diffusions = [get_diffusion_coefficient(radius) for radius in radii]
    noisy, gt, paths = generate_multiparticle_kymograph(
        length=512, width=512,
        diffusion=diffusions,
        contrast=[0.8, 0.65],
        noise_level=0.15,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=42,
    )
    cases.append({
        'name': 'Two Particles (Low Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 2: Two particles, high noise
    print("Generating case 2: Two particles, high noise...")
    radii = [12.0, 24.0]
    diffusions = [get_diffusion_coefficient(radius) for radius in radii]
    noisy, gt, paths = generate_multiparticle_kymograph(
        length=512, width=512,
        diffusion=diffusions,
        contrast=[0.6, 0.45],
        noise_level=0.4,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=43,
    )
    cases.append({
        'name': 'Two Particles (High Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 3: Two particles, moderate noise
    print("Generating case 3: Two particles, moderate noise...")
    radius1, radius2 = 8.0, 20.0
    diffusion1 = get_diffusion_coefficient(radius1)
    diffusion2 = get_diffusion_coefficient(radius2)
    noisy, gt, paths = generate_multiparticle_kymograph(
        length=512, width=512,
        diffusion=[diffusion1, diffusion2],
        contrast=[0.7, 0.5],
        noise_level=0.3,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=44,
    )
    cases.append({
        'name': 'Two Particles (Moderate Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 4: Three particles, moderate noise
    print("Generating case 4: Three particles, moderate noise...")
    radii = [5.0, 12.0, 25.0]
    diffusions = [get_diffusion_coefficient(r) for r in radii]
    noisy, gt, paths = generate_multiparticle_kymograph(
        length=512, width=512,
        diffusion=diffusions,
        contrast=[0.8, 0.6, 0.5],
        noise_level=0.25,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=45,
    )
    cases.append({
        'name': 'Three Particles (Moderate Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 5: Two particles, high noise
    print("Generating case 5: Two particles, high noise...")
    radius1, radius2 = 10.0, 18.0
    diffusion1 = get_diffusion_coefficient(radius1)
    diffusion2 = get_diffusion_coefficient(radius2)
    noisy, gt, paths = generate_multiparticle_kymograph(
        length=512, width=512,
        diffusion=[diffusion1, diffusion2],
        contrast=[0.5, 0.4],
        noise_level=0.5,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=46,
    )
    cases.append({
        'name': 'Two Particles (High Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    return cases


def main():
    """Main demo function."""
    print("=" * 70)
    print("KYMO-TRACKER DEMO: Classical vs Deep Learning Comparison")
    print("=" * 70)
    
    # Setup paths
    demo_dir = Path(__file__).parent
    model_path = demo_dir.parent / "artifacts" / "demo_model.pth"
    checkpoint_dir = demo_dir.parent / "checkpoints"
    output_dir = demo_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Step 1: Train model if it doesn't exist
    if not model_path.exists():
        print("\n" + "=" * 70)
        print("STEP 1: Training model (this may take a while)...")
        print("=" * 70)
        
        # Lightweight training for demo
        dataset = MultiTaskDataset(
            n_samples=1024,  # Reduced for faster training
            window_length=DEFAULT_WINDOW_LENGTH,
            length=DEFAULT_WIDTH,
            width=DEFAULT_WIDTH,
            radii_nm=DEFAULT_RADII_NM,
            contrast=DEFAULT_CONTRAST,
            noise_level=DEFAULT_NOISE_LEVEL,
            min_trajectories=DEFAULT_MIN_TRAJECTORIES,
            max_trajectories=DEFAULT_MAX_TRAJECTORIES,
            mask_peak_width_samples=DEFAULT_MASK_PEAK_WIDTH_SAMPLES,
        )
        
        config = MultiTaskConfig(
            epochs=DEFAULT_EPOCHS,
            batch_size=16,
            learning_rate=1.5e-3,
            checkpoint_dir=str(checkpoint_dir),
            auto_resume=False,  # Don't resume for demo
        )
        
        model = train_multitask_model(config, dataset)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_multitask_model(model, str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print(f"\nModel found at {model_path}, skipping training...")
    
    # Step 2: Load model
    print("\n" + "=" * 70)
    print("STEP 2: Loading model...")
    print("=" * 70)
    model = load_multitask_model(str(model_path), device=device, max_tracks=3)
    print("Model loaded successfully")
    
    # Step 3: Generate test cases
    print("\n" + "=" * 70)
    print("STEP 3: Generating test cases...")
    print("=" * 70)
    test_cases = generate_demo_cases()
    
    # Step 4: Run inference and create plots
    print("\n" + "=" * 70)
    print("STEP 4: Running inference and creating comparison plots...")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nProcessing case {i}/5: {case['name']}...")
        
        noisy_kymo = case['noisy']
        true_paths = case.get('true_paths', None)
        
        # Run classical pipeline
        print("  Running classical pipeline...")
        classical_result = run_classical_pipeline(noisy_kymo)
        
        # Run deep learning pipeline
        print("  Running deep learning pipeline...")
        dl_result = run_deeplearning_pipeline(noisy_kymo, model, device)
        
        # Create comparison plot
        print("  Creating comparison plot...")
        output_path = output_dir / f"comparison_case_{i:02d}_{case['name'].replace(' ', '_').lower()}.png"
        visualize_comparison(
            noisy_kymo=noisy_kymo,
            classical_filtered=classical_result['filtered'],
            classical_mask=classical_result['mask'],
            classical_trajectories=classical_result['trajectories'],
            deeplearning_denoised=dl_result['denoised'],
            deeplearning_heatmap=dl_result['heatmap'],
            deeplearning_trajectories=dl_result['trajectories'],
            true_paths=true_paths,
            output_path=str(output_path),
            title=f"Case {i}: {case['name']}",
        )
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"All comparison plots saved to: {output_dir}")
    print(f"Generated {len(test_cases)} comparison plots")


if __name__ == "__main__":
    main()
