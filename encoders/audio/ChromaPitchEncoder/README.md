# ChromaPitchEncoder

`ChromaPitchEncoder` is based on chroma spectrograms (chromagrams) which represent melodic/harmonic features.
    
`ChromaPitchEncoder` encodes an audio signal from a `Batch x Signal Length` ndarray into a
    `Batch x Concatenated Features` ndarray.