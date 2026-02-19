import logging
import os
from pathlib import Path
import urllib
import warnings
from typing import Optional

import torch
import torchopenl3  # Check https://github.com/torchopenl3/torchopenl3
from einops import rearrange
from filelock import FileLock, Timeout

LOGGER = logging.getLogger(__name__)
CACHE_SUBFOLDER_NAME = Path("weights")
MODEL_FILENAME_TEMPLATE = "torchopenl3_$repr$_$cont$_$size$.pth.tar"


###################################################################################################


class OpenL3Frontend(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        input_representation: str = "mel256",  # One of (linear, mel128, mel256)
        content_type: str = "env",  # One of (env, music)
        embedding_size: int = 512,  # One of (512, 6144)
        hop_size: float = 0.5,  # Frame size set by torchopenl3 to 1 sec
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        # Model filename
        model_filename = MODEL_FILENAME_TEMPLATE.replace("$repr$", input_representation)
        model_filename = model_filename.replace("$cont$", content_type)
        model_filename = model_filename.replace("$size$", str(embedding_size))
        model_path = CACHE_SUBFOLDER_NAME / model_filename.replace(".pth.tar", "")
        model_filename = model_path / model_filename
        # Need to download model?

        if not os.path.exists(model_filename):
            LOGGER.info(f"OpenL3 model not found: downloading {model_filename}")

            # Create dir
            model_path.mkdir(parents=True, exist_ok=True)

            lock_filename = str(model_filename) + ".lock"
            lock = FileLock(lock_filename)
            try:
                with lock.acquire(timeout=1000):
                    if os.path.exists(model_filename):
                        print(
                            f"File {model_filename} already exists. Skipping download."
                        )
                    else:
                        # Do download from the internet
                        url = torchopenl3.core.get_model_path(
                            input_representation, content_type, embedding_size
                        )
                        urllib.request.urlretrieve(url, model_filename)
            except Timeout:
                LOGGER.info("Another process is downloading the model - waiting")

            # Signal download is done

        # Load model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torchopenl3.models.PytorchOpenl3(
                input_repr=input_representation,
                content_type=content_type,
                embedding_size=embedding_size,
            )
        self.model.load_state_dict(torch.load(model_filename))
        self.eval()
        # Init
        self.minlen = int(1.6 * self.sample_rate)

    def ensure_min_length(self, x):
        if x.size(-1) < self.minlen:
            xorig = x.clone()
            while x.size(-1) < self.minlen:
                x = torch.cat([x, xorig], dim=-1)
            x = x[..., : self.minlen]
        return x

    def forward(
        self,
        x: torch.Tensor,  # (B,T)
        intermediate_features: bool = False,  # Get intermediate activations?
    ):
        assert x.ndim == 2
        x = self.ensure_min_length(x)
        # Resample if required (julius) and ensure minimum length
        y = torchopenl3.utils.preprocess_audio_batch(
            x, self.sample_rate, hop_size=self.hop_size
        )
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute features and reshape
            z = self.model(y.float(), keep_all_outputs=intermediate_features)
        if intermediate_features:
            # Select all activations after max pooling & reshape
            z = [z[8], z[15], z[22], z[27]]
            for i in range(len(z)):
                z[i] = rearrange(z[i], "(b n) ... -> b ... n", b=x.size(0))
        else:
            # Reshape
            z = rearrange(z, "(b n) c -> b c n", b=x.size(0))
        return z


###################################################################################################
