import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/trust/vae_trust.py
"""
Variational Autoencoder-based trust scoring.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class VAETrust(BaseTrustModel):
    """
    Trust scoring using VAE reconstruction error.

    Assumes reference data follows learned distribution;
    high reconstruction error indicates unfamiliar samples.

    Parameters
    ----------
    latent_dim : int, default=32
        VAE latent dimension.
    hidden_dims : list, default=[128, 64]
        Hidden layer dimensions.
    epochs : int, default=100
        Training epochs.
    batch_size : int, default=256
        Training batch size.
    threshold_percentile : float, default=95
        Percentile for anomaly threshold.

    Examples
    --------
    >>> trust_model = VAETrust(latent_dim=16)
    >>> trust_model.fit(X_reference)
    >>> scores = trust_model.score(X_test)
    """

    def __init__(
        self,
        name: str = "VAETrust",
        calibration_method: str = "isotonic",
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        epochs: int = 100,
        batch_size: int = 256,
        threshold_percentile: float = 95,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.learning_rate = learning_rate

        self.vae_ = None
        self.threshold_ = None
        self.input_dim_ = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "VAETrust":
        """Train VAE on reference data."""
        X = np.asarray(X).astype(np.float32)
        self.input_dim_ = X.shape[1]

        # Detect framework
        if self._is_torch_available():
            self._fit_torch(X)
        elif self._is_tf_available():
            self._fit_tensorflow(X)
        else:
            raise ImportError("PyTorch or TensorFlow required for VAE")

        self.is_fitted_ = True
        return self

    def _is_torch_available(self) -> bool:
        try:
            import torch

            return True
        except ImportError:
            return False

    def _is_tf_available(self) -> bool:
        try:
            import tensorflow as tf

            return True
        except ImportError:
            return False

    def _fit_torch(self, X: npt.NDArray):
        """PyTorch VAE implementation."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dims):
                super().__init__()
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    encoder_layers.append(nn.Linear(prev_dim, h_dim))
                    encoder_layers.append(nn.ReLU())
                    prev_dim = h_dim
                self.encoder = nn.Sequential(*encoder_layers)
                self.fc_mu = nn.Linear(prev_dim, latent_dim)
                self.fc_var = nn.Linear(prev_dim, latent_dim)

                # Decoder
                decoder_layers = []
                prev_dim = latent_dim
                for h_dim in reversed(hidden_dims):
                    decoder_layers.append(nn.Linear(prev_dim, h_dim))
                    decoder_layers.append(nn.ReLU())
                    prev_dim = h_dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_var(h)

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                return self.decode(z), mu, log_var

        # Initialize
        self.vae_ = VAE(self.input_dim_, self.latent_dim, self.hidden_dims)
        optimizer = optim.Adam(self.vae_.parameters(), lr=self.learning_rate)

        # Training
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.vae_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in dataloader:
                optimizer.zero_grad()
                recon, mu, log_var = self.vae_(batch)

                # Reconstruction + KL loss
                recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")

        # Compute threshold
        self.vae_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X)
            recon, _, _ = self.vae_(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
            self.threshold_ = np.percentile(errors, self.threshold_percentile)

    def _fit_tensorflow(self, X: npt.NDArray):
        """TensorFlow VAE implementation."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Encoder
        inputs = keras.Input(shape=(self.input_dim_,))
        x = inputs
        for h_dim in self.hidden_dims:
            x = layers.Dense(h_dim, activation="relu")(x)

        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)

        # Sampling
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # Decoder
        decoder_input = keras.Input(shape=(self.latent_dim,))
        x = decoder_input
        for h_dim in reversed(self.hidden_dims):
            x = layers.Dense(h_dim, activation="relu")(x)
        outputs = layers.Dense(self.input_dim_)(x)

        decoder = keras.Model(decoder_input, outputs)
        encoder = keras.Model(inputs, [z_mean, z_log_var, z])

        # Full VAE
        vae_outputs = decoder(encoder(inputs)[2])
        self.vae_ = keras.Model(inputs, vae_outputs)

        # Loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(inputs - vae_outputs), axis=-1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        self.vae_.add_loss(reconstruction_loss + kl_loss)

        self.vae_.compile(optimizer="adam")
        self.vae_.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        # Compute threshold
        reconstructions = self.vae_.predict(X)
        errors = np.mean((X - reconstructions) ** 2, axis=1)
        self.threshold_ = np.percentile(errors, self.threshold_percentile)

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust scores from reconstruction error.

        Returns
        -------
        scores : ndarray
            Trust scores in [0, 1], higher = more trustworthy.
        """
        self._check_is_fitted()
        X = np.asarray(X).astype(np.float32)

        if self._is_torch_available() and isinstance(self.vae_, torch.nn.Module):
            import torch

            self.vae_.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X)
                recon, _, _ = self.vae_(X_tensor)
                errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
        else:
            reconstructions = self.vae_.predict(X)
            errors = np.mean((X - reconstructions) ** 2, axis=1)

        # Convert to trust score (exponential decay from threshold)
        scores = np.exp(-errors / (self.threshold_ + 1e-8))
        return np.clip(scores, 0, 1)

    def get_reconstruction(self, X: npt.ArrayLike) -> npt.NDArray:
        """Get VAE reconstruction for visualization."""
        self._check_is_fitted()
        X = np.asarray(X).astype(np.float32)

        if self._is_torch_available():
            import torch

            self.vae_.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X)
                recon, _, _ = self.vae_(X_tensor)
                return recon.numpy()
        else:
            return self.vae_.predict(X)
