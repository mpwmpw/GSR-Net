import math
import random
import numpy as np
from ultralytics.utils import LOGGER, yaml_save

class RPSOTuner:
    """
    R-PSO (Randomized Particle Swarm Optimization) tuner for hyperparameter optimization.
    Replaces GA-based tuner while keeping Ultralytics logging and result saving.
    """

    def __init__(self, space, n_particles=10, inertia=0.7, cognitive=1.5, social=1.5,
                 iters=50, save_dir="runs/tune"):
        """
        Args:
            space (dict): hyperparameter search space, e.g. {'lr0': (1e-5, 1e-1), 'momentum': (0.6, 0.98)}
            n_particles (int): number of particles in the swarm
            inertia (float): inertia weight
            cognitive (float): cognitive coefficient
            social (float): social coefficient
            iters (int): number of iterations
            save_dir (str): directory for saving results
        """
        self.space = space
        self.params = list(space.keys())
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.iters = iters
        self.save_dir = save_dir

        # Initialize particles, velocities, and bests
        self.particles = [self._random_params() for _ in range(n_particles)]
        self.velocities = [np.zeros(len(self.params)) for _ in range(n_particles)]
        self.personal_best = list(self.particles)
        self.personal_best_score = [-math.inf] * n_particles
        self.global_best = None
        self.global_best_score = -math.inf

    def _random_params(self):
        """Initialize particle with random parameters inside the search space."""
        return np.array([
            random.uniform(self.space[p][0], self.space[p][1])
            for p in self.params
        ])

    def _evaluate(self, params, evaluate_fn):
        """Call external evaluation function (e.g. YOLO training + validation)."""
        hp_dict = {k: float(v) for k, v in zip(self.params, params)}
        score = evaluate_fn(hp_dict)
        return score

    def __call__(self, evaluate_fn):
        """Run the R-PSO optimization loop."""
        LOGGER.info(f"Starting R-PSO tuning with {self.n_particles} particles for {self.iters} iterations")

        for it in range(self.iters):
            LOGGER.info(f"\nIteration {it+1}/{self.iters}")

            for i in range(self.n_particles):
                # Evaluate particle
                score = self._evaluate(self.particles[i], evaluate_fn)

                # Update personal best
                if score > self.personal_best_score[i]:
                    self.personal_best_score[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

                LOGGER.info(f"Particle {i+1}/{self.n_particles} "
                            f"score={score:.4f} best={self.global_best_score:.4f}")

            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                inertia_term = self.inertia * self.velocities[i]
                cognitive_term = self.cognitive * r1 * (self.personal_best[i] - self.particles[i])
                social_term = self.social * r2 * (self.global_best - self.particles[i])
                random_term = np.random.normal(0, 0.01, size=len(self.params))  # random noise for R-PSO

                self.velocities[i] = inertia_term + cognitive_term + social_term + random_term
                self.particles[i] += self.velocities[i]

                # Boundary clipping
                for j, p in enumerate(self.params):
                    low, high = self.space[p]
                    self.particles[i][j] = np.clip(self.particles[i][j], low, high)

        # Save best hyperparameters
        best_hp = {k: float(v) for k, v in zip(self.params, self.global_best)}
        yaml_save(f"{self.save_dir}/best_hyperparameters.yaml", best_hp)
        LOGGER.info(f"\nBest hyperparameters saved to {self.save_dir}/best_hyperparameters.yaml")
        LOGGER.info(f"Best score: {self.global_best_score:.4f}")
        return best_hp
