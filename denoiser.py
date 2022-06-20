import sys
import torch
from stft import STFT


class Denoiser(torch.nn.Module):
    """ WaveGlow denoiser, adapted for HiFi-GAN """

    def __init__(
        self, hifigan, filter_length=1024, n_overlap=4, win_length=1024, mode="zeros"
    ):
        if torch.cuda.is_available() == True:
            cpuorgpu = True
        else:
            cpuorgpu = False
        super(Denoiser, self).__init__()
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
        )
        if cpuorgpu:
            self.stft.cuda()
        else:
            self.stft.cpu()
        if mode == "zeros":
            mel_input = torch.zeros((1, 80, 88))
            if cpuorgpu:
                mel_input.cuda()
            else:
                mel_input.cpu()
        elif mode == "normal":
            mel_input = torch.randn((1, 80, 88))
            if cpuorgpu:
                mel_input.cuda()
            else:
                mel_input.cpu()
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = hifigan(mel_input).view(1, -1).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
    
        if torch.cuda.is_available() == True:
            cpuorgpu = True
        else:
            cpuorgpu = False
         
        if cpuorgpu:
            audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        else:
            audio_spec, audio_angles = self.stft.transform(audio.cpu().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
