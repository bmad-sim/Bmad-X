from collections import namedtuple

Particle = namedtuple('Particle', 'x px y py z pz s p0c mc2')

Drift = namedtuple('Drift', 'L')

Quadrupole = namedtuple('Quadrupole',
                        [
                            'L',
                            'K1',
                            'NUM_STEPS',
                            'X_OFFSET',
                            'Y_OFFSET',
                            'TILT'
                        ],
                        defaults=(None, None, 1, 0, 0, 0))

CrabCavity = namedtuple('CrabCavity', 
                        [
                            'L', 
                            'VOLTAGE',
                            'PHI0',
                            'RF_FREQUENCY',
                            'X_OFFSET',
                            'Y_OFFSET',
                            'TILT'
                        ],
                        defaults=(None, None, 1, 0, 0, 0))

RFCavity = namedtuple('RFCavity', 
                      [
                          'L', 
                          'VOLTAGE',
                          'PHI0',
                          'RF_FREQUENCY',
                          'X_OFFSET',
                          'Y_OFFSET',
                          'TILT'
                      ],
                      defaults=(None, None, 1, 0, 0, 0))