# MFCCTimbreEncoder

`MFCCTimbreEncoder` is based on Mel-Frequency Cepstral Coefficients (MFCCs) which represent timbral features.
    
`MFCCTimbreEncoder` encodes an audio signal from a `Batch x Signal Length` ndarray into a
    `Batch x Concatenated Features` ndarray.