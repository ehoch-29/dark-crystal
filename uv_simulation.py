import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, binom
from scipy.special import beta as beta_function

# Parameters
width, height = 512, 512
wavelength = 250 # nm
mean_incident_energy_per_pixel = 1239.8/wavelength  # eV, energy deposited per pixel

# Silicon parameters at 300 K (from paper)
E_g = 1.124  # Band gap energy (eV)
hbar_omega_0 = 0.063  # Optical phonon energy (eV)
A = 5.2  # Phonon-to-ionization scattering ratio (eV^2)
W = 12.0  # Valence band width (eV)
hbar_omega_pl = 16.6  # Plasmon energy (eV)

def beta_distribution(x, alpha):
    """Beta distribution for energy partitioning between electron and hole."""
    if alpha <= 0:
        return np.inf  # Dirac delta behavior
    return (2 / beta_function(alpha, alpha)) * (x ** (alpha - 1)) * ((1 - x) ** (alpha - 1))

def impact_ionization_probability(E, A_param):
    """Probability of impact ionization vs phonon emission (Eq. 10)."""
    if E <= 0:
        return 0.0
    rate_ratio = A_param * (1e5 / (2 * np.pi)) * ((E - hbar_omega_0) ** 0.5) / ((E - E_g) ** 3.5)
    return 1.0 / (1.0 + rate_ratio)

def monte_carlo_charge_yield(E_r, num_trials=1000, alpha_energy=1.0):
    """
    Monte Carlo simulation of charge yield following Ramanathan & Kurinsky (2004.10709).
    
    Args:
        E_r: Recoil energy (eV)
        num_trials: Number of cascade simulations
        alpha_energy: Beta distribution shape parameter for energy partitioning
    
    Returns:
        Distribution of electron-hole pairs created
    """
    charge_distributions = []
    
    for _ in range(num_trials):
        if E_r < E_g:
            charge_distributions.append(0)
            continue
        
        particles = []  # (energy, is_electron)
        
        # Initial energy partitioning between electron and hole
        x = np.random.beta(alpha_energy, alpha_energy)
        E_e = x * (E_r - E_g)
        E_h = (1 - x) * (E_r - E_g)
        
        # Enforce valence band constraint
        if E_h > W:
            E_e += E_h - W
            E_h = W
        
        particles.append((E_e, True))   # electron
        particles.append((E_h, False))  # hole
        
        n_pairs = 1  # Initial pair
        
        # Cascade process
        while particles:
            E_current, is_electron = particles.pop(0)
            
            # Handle plasmon creation for high-energy carriers
            if E_current > hbar_omega_pl and is_electron:
                n_plasmons = int(E_current / hbar_omega_pl)
                E_current -= n_plasmons * hbar_omega_pl
                # Treat plasmons as creating additional e-h pairs
                n_pairs += n_plasmons
            
            # Cascade down through impact ionization or phonon emission
            while E_current > E_g:
                p_ionize = impact_ionization_probability(E_current, A)
                
                if np.random.random() < p_ionize:
                    # Impact ionization occurs
                    # Energy split between original carrier and new e-h pair
                    E_new_pair = E_current - E_g
                    x_split = np.random.beta(1, 1)  # Uniform split for cascade
                    E_e_new = x_split * E_new_pair
                    E_h_new = (1 - x_split) * E_new_pair
                    
                    # Apply valence band constraint
                    if E_h_new > W:
                        E_e_new += E_h_new - W
                        E_h_new = W
                    
                    E_current = E_e_new
                    particles.append((E_h_new, False))
                    n_pairs += 1
                else:
                    # Phonon emission
                    E_current -= hbar_omega_0
        
        charge_distributions.append(n_pairs)
    
    return np.array(charge_distributions)

# Simulate CCD image
print("Simulating CCD image with Ramanathan & Kurinsky ionization model...")
photoelectrons = np.zeros((height, width), dtype=int)

for i in range(height):
    if i % 50 == 0:
        print(f"  Processing row {i}/{height}...")
    for j in range(width):
        # Poisson-distributed incident energy
        #E_incident = poisson.rvs(mu = mean_incident_energy_per_pixel)
        E_incident = np.random.normal(mean_incident_energy_per_pixel, 0.04)

        if E_incident > 0:
            # Use alpha=1 (uniform energy distribution) for simplicity
            # For more accuracy, use energy-dependent alpha from Fig. 4
            charge_dist = monte_carlo_charge_yield(E_incident, num_trials=5, alpha_energy=1.0)
            photoelectrons[i, j] = charge_dist[0]
        else:
            photoelectrons[i, j] = 0

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: CCD Image
im = axes[0].imshow(photoelectrons, cmap='viridis', origin='upper')
axes[0].set_title('Simulated CCD Image (512 x 512)\nwith Ionization Yield Model', fontsize=12)
axes[0].set_xlabel('X Pixel')
axes[0].set_ylabel('Y Pixel')
cbar = plt.colorbar(im, ax=axes[0])
cbar.set_label('Photoelectrons', rotation=270, labelpad=15)

# Plot 2: Histogram of photoelectron counts
n, bins, patches  = axes[1].hist(photoelectrons.flatten(), bins=np.arange(0, photoelectrons.max() + 2) - 0.5, 
             edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution of Photoelectrons per Pixel', fontsize=12)
axes[1].set_xlabel('Photoelectrons per Pixel')
axes[1].set_ylabel('Number of Pixels')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

# Print statistics
print(f"\nImage Statistics:")
print(f"  Mean incident energy per pixel: {mean_incident_energy_per_pixel} eV")
print(f"  Mean photoelectrons per pixel: {photoelectrons.mean():.2f}")
print(f"  Standard deviation: {photoelectrons.std():.2f}")
print(f"  Count Distribution: {n}")
print(f"  Min photoelectrons: {photoelectrons.min()}")
print(f"  Max photoelectrons: {photoelectrons.max()}")
print(f"\nSilicon Parameters (300 K):")
print(f"  Band gap E_g: {E_g} eV")
print(f"  Optical phonon: {hbar_omega_0*1000:.1f} meV")
print(f"  Valence band width W: {W} eV")
print(f"  Scattering ratio A: {A} eV^2")


