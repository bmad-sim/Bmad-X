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
                        defaults=(None, None, 1, 0.0, 0.0, 0.0)
                       )

Sextupole = namedtuple('Sextupole',
                        [
                            'L',
                            'K2',
                            'NUM_STEPS',
                            'X_OFFSET',
                            'Y_OFFSET',
                            'TILT'
                        ],
                        defaults=(None, None, 1, 0.0, 0.0, 0.0)
                       )

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
                        defaults=(None, None, 1, 0.0, 0.0, 0.0)
                       )

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
                      defaults=(None, None, 1, 0.0, 0.0, 0.0)
                     )

SBend = namedtuple('SBend', 
                   [
                       'L',
                       'P0C',
                       'G',
                       'DG',
                       'E1',
                       'E2',
                       'F_INT',
                       'H_GAP',
                       'F_INT_X',
                       'H_GAP_X',
                       'FRINGE_AT',
                       'FRINGE_TYPE'
                   ],
                   defaults=(None,
                             None, 
                             None, 
                             0.0, 
                             0.0, 
                             0.0, 
                             0.0, 
                             0.0, 
                             0.0, 
                             0.0, 
                             "both_ends", 
                             "none"
                            )
                  )