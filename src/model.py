# ============================================================
# model.py
# GeoPINONet — Neural network architecture
# ============================================================

import torch
import torch.nn as nn
from .config import LATENT_DIM, N_FOURIER_FEATURES, FOURIER_SCALES


# ============================================================
# POSITIONAL ENCODER
# ============================================================

class PositionalEncoder(nn.Module):
    """Multi-scale random Fourier feature encoder: (x, y, z) → [sin, cos] features."""

    def __init__(self, inputdims: int = 3, numfreqs: int = 256,
                 scales: list = None):
        super().__init__()
        if scales is None:
            scales = [1.0, 4.0, 8.0, 16.0]
        freqs_per_scale = numfreqs // len(scales)
        actual_freqs    = freqs_per_scale * len(scales)
        B = torch.cat([
            torch.randn(inputdims, freqs_per_scale, dtype=torch.float32) * s
            for s in scales
        ], dim=1)
        self.register_buffer('B', B)
        self.output_dim      = actual_freqs * 2
        self.freqs_per_scale = freqs_per_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x.to(self.B.dtype), self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward_low_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Only lowest-frequency scale (first freqs_per_scale columns of B)."""
        B_low = self.B[:, :self.freqs_per_scale]
        proj  = torch.matmul(x.to(self.B.dtype), B_low)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ============================================================
# POINTNET ENCODER
# ============================================================

class PointNetEncoder(nn.Module):
    """PointNet-based geometry encoder: point cloud → fixed-size latent vector."""

    def __init__(self, input_dim: int = 3, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(input_dim, 64,   1)
        self.conv2 = nn.Conv1d(64,        128,  1)
        self.conv3 = nn.Conv1d(128,       256,  1)
        self.conv4 = nn.Conv1d(256,       512,  1)
        self.conv5 = nn.Conv1d(512,       1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        x = point_cloud.unsqueeze(0).transpose(2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2)[0]
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(0)


# ============================================================
# PHYSICS DECODER
# ============================================================

class PhysicsDecoder(nn.Module):
    """
    MLP decoder that maps (positional features + latent vector) to
    displacement and stress field outputs via thickness-mode parameterization.

    Output layout (11 channels):
      0: UX   1: UY   2: SXX  3: SYY  4: SXY
      5: A_UZ  6: B_UZ        →  UZ  = A·z̃ + B·z̃³
      7: A_SYZ 8: B_SYZ       →  SYZ = A·z̃·(1-z̃²) + B·z̃³·(1-z̃²)
      9: A_SXZ                →  SXZ = A·z̃·(1-z̃²)
     10: B_SZZ                →  SZZ = B·(1 - z̃²)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        freqs_per_scale = N_FOURIER_FEATURES // len(FOURIER_SCALES)
        fourier_full    = freqs_per_scale * len(FOURIER_SCALES) * 2
        input_full      = fourier_full + latent_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_full, 768), nn.SiLU(),
            nn.Linear(768,        768), nn.SiLU(),
        )
        self.head_main = nn.Sequential(
            nn.Linear(768, 768), nn.SiLU(),
            nn.Linear(768, 768), nn.SiLU(),
            nn.Linear(768, 768), nn.SiLU(),
            nn.Linear(768, 11),
        )
        self._init_weights()

    def _init_weights(self):
        for m in list(self.trunk.modules()) + list(self.head_main.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords_full: torch.Tensor,
                latent_vector: torch.Tensor) -> torch.Tensor:
        latent_exp = latent_vector.unsqueeze(0).expand(coords_full.shape[0], -1)
        trunk_out  = self.trunk(torch.cat([coords_full, latent_exp], dim=1))
        return self.head_main(trunk_out)


# ============================================================
# GEOPINO NET
# ============================================================

class GeoPINONet(nn.Module):
    """
    Geometry-informed Physics-Informed Neural Operator for 3D structural mechanics.
    Maintains two independent encoder-decoder branches, one per load case:
      - COMP: axial compression
      - LAT:  lateral bending
    """

    def __init__(self):
        super().__init__()
        self.geometric_encoder_comp  = PointNetEncoder()
        self.positional_encoder_comp = PositionalEncoder(3, N_FOURIER_FEATURES, FOURIER_SCALES)
        self.decoder_comp            = PhysicsDecoder()

        self.geometric_encoder_lat   = PointNetEncoder()
        self.positional_encoder_lat  = PositionalEncoder(3, N_FOURIER_FEATURES, FOURIER_SCALES)
        self.decoder_lat             = PhysicsDecoder()

        self.log_vars = nn.Parameter(torch.zeros(12, dtype=torch.float32))

    def encode_geometry_comp(self, point_cloud: torch.Tensor) -> torch.Tensor:
        return self.geometric_encoder_comp(point_cloud)

    def encode_geometry_lat(self, point_cloud: torch.Tensor) -> torch.Tensor:
        return self.geometric_encoder_lat(point_cloud)

    def _forward(self, encoder: nn.Module, decoder: nn.Module,
                 latent: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                 z: torch.Tensor, z_mid: float, z_half: float):
        coords   = torch.cat([x, y, z], dim=1).float()
        coords_f = encoder(coords)
        main_out = decoder(coords_f, latent)

        z_tilde    = (z - z_mid) / z_half
        z_tilde3   = z_tilde ** 3
        shape_free = (1.0 - z_tilde ** 2)

        UX  = main_out[:, 0:1]
        UY  = main_out[:, 1:2]
        SXX = main_out[:, 2:3]
        SYY = main_out[:, 3:4]
        SXY = main_out[:, 4:5]

        UZ  = main_out[:, 5:6] * z_tilde + main_out[:, 6:7] * z_tilde3
        SYZ = (main_out[:, 7:8] * z_tilde * shape_free
               + main_out[:, 8:9] * z_tilde3 * shape_free)
        SXZ = main_out[:, 9:10]  * z_tilde * shape_free
        SZZ = main_out[:, 10:11] * shape_free

        u     = torch.cat([UX, UY, UZ],         dim=1)
        sigma = torch.cat([SXX, SYY, SZZ, SXY, SYZ, SXZ], dim=1)
        return u, sigma

    def forward_comp(self, latent: torch.Tensor, x: torch.Tensor,
                     y: torch.Tensor, z: torch.Tensor,
                     z_mid: float, z_half: float):
        return self._forward(self.positional_encoder_comp, self.decoder_comp,
                             latent, x, y, z, z_mid, z_half)

    def forward_lat(self, latent: torch.Tensor, x: torch.Tensor,
                    y: torch.Tensor, z: torch.Tensor,
                    z_mid: float, z_half: float):
        return self._forward(self.positional_encoder_lat, self.decoder_lat,
                             latent, x, y, z, z_mid, z_half)