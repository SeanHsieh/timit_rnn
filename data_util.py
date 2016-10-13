import glob
import os
import random

import essentia
from essentia.standard import MonoLoader, FrameGenerator, Resample, Windowing, Spectrum, MFCC, MonoWriter
import numpy as np

def mfcc_init(nfft=1024, nbands=40, ncoeffs=13, fs=44100.0):
  window = Windowing(size=nfft, type='blackmanharris62')
  spectrum = Spectrum(size=nfft)
  nfft_out = (nfft // 2) + 1
  mfcc = MFCC(inputSize=nfft_out,
              numberBands=nbands,
              numberCoefficients=ncoeffs,
              sampleRate=fs)
  return lambda frame: mfcc(spectrum(window(frame)))[1]

def load_timit(timit_dir,
               nfft=1024,
               nhop=512,
               limit=None,
               extractor=None):
  wrd_fps = glob.glob(os.path.join(timit_dir, '**/**/*.WRD'))
  limit = limit if limit else len(wrd_fps)
  resampler = Resample(inputSampleRate=16000, outputSampleRate=22000, quality=0)
  results = []
  for wrd_fp in wrd_fps[:limit]:
    wav_fp = '{}.WAV'.format(os.path.splitext(wrd_fp)[0])
    path_data = wrd_fp.split(os.sep)[-3:]
    dialect = path_data[0]
    sex = path_data[1][:1]
    speaker_id = path_data[1][1:]
    sentence_id = os.path.splitext(path_data[2])[0]
    metadata = (dialect, sex, speaker_id, sentence_id)

    loader = MonoLoader(filename=wav_fp, sampleRate=16000)
    utterance_22khz = resampler(loader())

    feats = []
    for frame in FrameGenerator(utterance_22khz, nfft, nhop):
      frame_feats = frame
      if extractor:
        frame_feats = extractor(frame)
      feats.append(frame_feats)

    results.append((metadata, feats))

  return results

if __name__ == '__main__':
  timit_train = '/media/cdonahue/bulk1/datasets/timit/TIMIT/TRAIN'
  timit_test = '/media/cdonahue/bulk1/datasets/timit/TIMIT/TEST'
  nfft = 512
  nhop = 256
  nbands = 40
  ncoeffs = 13
  mfcc_extractor = mfcc_init(nfft=nfft, nbands=nbands, ncoeffs=ncoeffs, fs=22000.0)
  timit_feats = load_timit(timit_train,
                           nfft=nfft,
                           nhop=nhop,
                           limit=16,
                           extractor=mfcc_extractor)
  import cPickle as pickle
  with open('timit_tiny_utter_mfcc_13.pkl', 'wb') as f:
    pickle.dump(timit_feats, f)
