
import torch
import torchaudio
import numpy as np

import dac
from audiotools import AudioSignal

# Load DAC model
model_path = 'pretrained/weights_16khz_8kbps_0.0.5.pth'
model = dac.DAC.load(model_path)
model.eval()

# Load audio signal file
audio_path = 'p227_001_16k.wav'
signal = AudioSignal(audio_path)
print('Load audio signal with {} length and {} sampling rate'.format(signal.audio_data.shape[-1], signal.sample_rate))

# Encode audio signal as one long file
signal.to(model.device)
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)
print('Audio length after processing: {}'.format(x.shape[-1]))

# Get audio tokens
codec_tokens = list(codes[0].T)
print('Get {} audio tokens with {} resigual layers'.format(len(codec_tokens), len(codec_tokens[0])))


class DacStreamingDecoder():
    def __init__(self,
                 model,
                 chunk_frames,
                 overlap_frames=None):
        self.model = model
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames if overlap_frames else self.compute_rf()
        self.code_buffer = []
        self.hop_length = np.prod(self.model.decoder_rates)

        assert self.chunk_frames > self.overlap_frames; \
            f'chunk_frames: {self.chunk_frames} should be larger than overlap_frames: {self.overlap_frames}'
        self.out_frames = self.chunk_frames - self.overlap_frames
        self.out_samples = self.out_frames * self.hop_length
        self.total_frames = 0

    def compute_rf(self, T=101):
        # create a random input latent feature
        latent_dim = self.model.latent_dim
        z = torch.randn(1, latent_dim, T).to(self.model.device)
        z.requires_grad_(True)
        z.retain_grad()
        out = self.model.decode(z)
        # backward
        grad = torch.zeros_like(out)
        grad[:, :, grad.shape[-1] // 2] = 1
        out.backward(grad)
        # select features that contain gradients
        gradmap = z.grad.detach().abs().sum(dim=1).squeeze(0)
        idx = (gradmap != 0).nonzero(as_tuple=False).squeeze(-1)
        center = T // 2
        left_rf_frames = center - int(idx.min().item())
        return left_rf_frames


    def streaming_decode(self,
                         token_frame: torch.Tensor):

        # Add token frames into buffer
        self.code_buffer.append(token_frame.detach().cpu())
        if len(self.code_buffer) < self.chunk_frames:
            return None

        # Build token chunk
        chunk = torch.stack(self.code_buffer[: self.chunk_frames], dim=0).transpose(0, 1).unsqueeze(0)  # (1, n_q, T)
        chunk = chunk.to(next(self.model.parameters()).device)

        # Decode
        with torch.no_grad():
            z, _, _ = self.model.quantizer.from_codes(chunk)
            y = self.model.decode(z)
        y = y.squeeze().cpu()
        out = y[:self.out_samples]
        self.code_buffer = self.code_buffer[self.out_frames: ]

        return out
    
    def decode_last(self):

        if len(self.code_buffer) == 0:
            return None
        
        chunk = torch.stack(self.code_buffer, dim=0).transpose(0, 1).unsqueeze(0)  # (1, n_q, T)
        chunk = chunk.to(next(self.model.parameters()).device)

        # Decode
        with torch.no_grad():
            z, _, _ = self.model.quantizer.from_codes(chunk)
            y = self.model.decode(z)
        y = y.squeeze().cpu()
        out = y
        return out

StreamingDACDecoder = DacStreamingDecoder(model, chunk_frames=20)
audio_wav_chunks = []
for token in codec_tokens:
    # Dac decodes as tokens become available.
    audio_wav_chunks.append(StreamingDACDecoder.streaming_decode(token))
audio_wav_chunks.append(StreamingDACDecoder.decode_last())
audio_wav_chunks = [c for c in audio_wav_chunks if c is not None]
audio_wav = torch.concat(audio_wav_chunks, dim=0)


audio_wav = audio_wav[:x.shape[-1]]
breakpoint()

