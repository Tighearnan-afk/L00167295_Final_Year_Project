import sklearn as ski
import jiwer as jw

psnr = ski.metrics.peak_signal_noise_ratio(y_true, y_pred)
ssim = ski.metrics.ssim(y_true, y_pred)
wer = jw.wer(y_true, y_pred)
cer = jw.cer(y_true, y_pred)